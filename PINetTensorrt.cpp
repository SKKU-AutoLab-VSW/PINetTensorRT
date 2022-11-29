#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <dirent.h>
#include <string.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

namespace {
    const std::string gSampleName = "TensorRT.onnx_PINet";

    const int output_base_index = 3;
    const float threshold_point = 0.81f;
    const float threshold_instance = 0.22f;
    const int resize_ratio = 8;

    int64 total_inference_execute_elasped_time = 0;
    int64 total_inference_execute_times = 0;

    using LaneLine = std::vector<cv::Point2f>;
    using LaneLines = std::vector<LaneLine>;

    cv::Mat chwDataToMat(int channelNum, int height, int width, float* data, cv::Mat& mask) {
        std::vector<cv::Mat> channels(channelNum);
        int data_size = width * height;
        for (int c = 0; c < channelNum; ++c) {
            float* channel_data = data + data_size * c;
            cv::Mat channel(height, width, CV_32FC1);
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w, ++channel_data) {
                    channel.at<float>(h, w) = *channel_data * (int)mask.at<uchar>(h, w);
                }
            }
            channels[c] = channel;
        }

        cv::Mat mergedMat;
        cv::merge(channels.data(), channelNum, mergedMat);
        return mergedMat;
    }

    void getFiles(std::string root_dir, std::string ext, std::vector<std::string>& files) {
        DIR *dir;
        struct dirent *ptr;

        if ((dir = opendir(root_dir.c_str())) == NULL) {
            sample::gLogInfo << "Open dir error..." << std::endl;
            return;
        }
    
        while ((ptr = readdir(dir)) != NULL) {
            if (strcmp(ptr->d_name,".") == 0 || strcmp(ptr->d_name,"..") == 0) {
                continue;
            } else if(ptr->d_type == 8)  {// file
                char* dot = strchr(ptr->d_name, '.');
                if (dot && !strcasecmp(dot, ext.c_str())) {
                    std::string filename(root_dir);
                    filename.append("/").append(ptr->d_name);
                    files.push_back(filename);
                }
            } else if(ptr->d_type == 10) { // link file  
                continue;
            } else if(ptr->d_type == 4)  {// dir
                std::string dir_path(root_dir);
                dir_path.append("/").append(ptr->d_name);
                getFiles(dir_path.c_str(), ext, files);
            }  
        }

        closedir(dir);  
    }
}

//! \brief  The PINetTensorrt class implements the ONNX PINet sample
//!
//! \details It creates the network using an ONNX model
//!
class PINetTensorrt
{
public:
    PINetTensorrt(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    void setImageFile(const std::string& imageFileName) {
        mImageFileName = imageFileName;
    }

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    std::vector<nvinfer1::Dims> mOutputDims; //!< The dimensions of the output to the network.
    std::string mImageFileName;            //!< The number to classify
    cv::Mat mInputImage;

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);
    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);

    void generatePostData(float* confidance_data, float* offsets_data, float* instance_data, cv::Mat& mask, cv::Mat& offsets, cv::Mat& features);

    LaneLines generateLaneLine(float* confidance_data, float* offsets_data, float* instance_data);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool PINetTensorrt::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        return false;
    }   

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
       return false;
    }

    if (sample::gLogger.getReportableSeverity() == sample::Logger::Severity::kVERBOSE) {
        for (int i = 0; i < network->getNbInputs(); ++i) {
            nvinfer1::Dims dim = network->getInput(i)->getDimensions();
            sample::gLogInfo << "InputDims: " << i << " " << dim.d[1] << " " << dim.d[2] << " " << dim.d[3] << std::endl;
        }

        for (int i = 0; i < network->getNbOutputs(); ++i) {
            nvinfer1::Dims dim = network->getOutput(i)->getDimensions();
            sample::gLogInfo << "OutputDims: " << i << " " << dim.d[1] << " " << dim.d[2] << " " << dim.d[3] << std::endl;
        }
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() == 6);
    for (int i = 0; i < network->getNbOutputs(); ++i) {
        nvinfer1::Dims dim = network->getOutput(i)->getDimensions();
        mOutputDims.push_back(dim);
        ASSERT(dim.nbDims == 4);
    }

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool PINetTensorrt::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(mParams.onnxFileName.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool PINetTensorrt::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    auto inferenceBeginTime = std::chrono::high_resolution_clock::now();
    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    auto inference_execute_elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - inferenceBeginTime);
    total_inference_execute_elasped_time += inference_execute_elapsed_time.count();
    ++total_inference_execute_times;

    //sample::gLogInfo << "inference elapsed time: " << inference_execute_elapsed_time.count() / 1000.f << " milliseconds" << std::endl;

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool PINetTensorrt::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputC = mInputDims.d[1];
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    cv::Mat image = cv::imread(mImageFileName, 1);
    assert(inputC == image.channels());
    cv::resize(image, image, cv::Size(inputW, inputH));

    mInputImage = image;

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    uchar* imageData = image.ptr<uchar>();
    for (int c = 0; c < inputC; ++c) {
        for (unsigned j = 0, volChl = inputW * inputH; j < volChl; ++j) {
            hostDataBuffer[c * volChl + j] = float(imageData[j * inputC + c]) / 255.f;
        }
    }

    return true;
}

void PINetTensorrt::generatePostData(float* confidance_data, float* offsets_data, float* instance_data, cv::Mat& mask, cv::Mat& offsets, cv::Mat& features)
{
    const nvinfer1::Dims& dim            = mOutputDims[output_base_index + 0];//1 32 64
    const nvinfer1::Dims& offset_dim     = mOutputDims[output_base_index + 1];//2 32 64
    const nvinfer1::Dims& instance_dim   = mOutputDims[output_base_index + 2];//4 32 64

    mask = cv::Mat::zeros(dim.d[2], dim.d[3], CV_8UC1);
    float* confidance_ptr = confidance_data;
    for (int i = 0; i < dim.d[2]; ++i) {
        for (int j = 0; j < dim.d[3]; ++j, ++confidance_ptr) {
            if (*confidance_ptr > threshold_point) {
                mask.at<uchar>(i, j) = 1;
            }
        }
    }

    if (sample::gLogger.getReportableSeverity() == sample::Logger::Severity::kVERBOSE) {
        sample::gLogInfo << "Output mask:" << std::endl;
        for (int i = 0; i < dim.d[2]; ++i) {
            for (int j = 0; j < dim.d[3]; ++j) {
                sample::gLogInfo << (int)mask.at<uchar>(i, j);
            }
            sample::gLogInfo << std::endl;
        }

        cv::Mat maskImage = mInputImage.clone();
        cv::Scalar color(0, 0, 255);
        for (int i = 0; i < dim.d[2]; ++i) {
            for (int j = 0; j < dim.d[3]; ++j) {
                if ((int)mask.at<uchar>(i, j)) {
                    cv::circle(maskImage, cv::Point2f(j * 8, i * 8), 3, color, -1);
                }
            }
        }
        cv::imshow("mask", maskImage);
        cv::waitKey(0);
    }

    offsets  = chwDataToMat(offset_dim.d[1], offset_dim.d[2], offset_dim.d[3], offsets_data, mask);
    features = chwDataToMat(instance_dim.d[1], instance_dim.d[2], instance_dim.d[3], instance_data, mask);    

    if (sample::gLogger.getReportableSeverity() == sample::Logger::Severity::kVERBOSE) {
        sample::gLogInfo << "Output offset:" << std::endl;
        for (int i = 0; i < dim.d[2]; ++i) {
            for (int j = 0; j < dim.d[3]; ++j) {
                sample::gLogInfo << (offsets.at<cv::Vec2f>(i, j)[0] ? 1 : 0);
            }
            sample::gLogInfo << std::endl;
        }

        cv::Mat offsetImage = mInputImage.clone();
        cv::Scalar color(0, 0, 255);
        for (int i = 0; i < dim.d[2]; ++i) {
            for (int j = 0; j < dim.d[3]; ++j) {
                if ((int)mask.at<uchar>(i, j)) {
                    cv::Vec2f pointOffset = offsets.at<cv::Vec2f>(i, j);
                    cv::Point2f point(pointOffset[1] + j, pointOffset[0] + i);
                    cv::circle(offsetImage, point * 8, 3, color, -1);
                }
            }
        }
        cv::imshow("offset", offsetImage);
        cv::waitKey(0);

        sample::gLogInfo << "Output instance:" << std::endl;
        for (int i = 0; i < dim.d[2]; ++i) {
            for (int j = 0; j < dim.d[3]; ++j) {
                sample::gLogInfo << (features.at<cv::Vec4f>(i, j)[0] ? 1 : 0);
            }
            sample::gLogInfo << std::endl;
        }
    }
}

LaneLines PINetTensorrt::generateLaneLine(float* confidance_data, float* offsets_data, float* instance_data)
{
    const nvinfer1::Dims& dim = mOutputDims[output_base_index];//1 32 64

    cv::Mat mask, offsets, features;
    generatePostData(confidance_data, offsets_data, instance_data, mask, offsets, features);
    
    LaneLines laneLines;
    std::vector<cv::Vec4f> laneFeatures;

    auto findNearestFeature = [&laneFeatures](const cv::Vec4f& feature) -> int {
        for (int i = 0; i < laneFeatures.size(); ++i) {
            auto delta = laneFeatures[i] - feature;
            if (delta.dot(delta) <= threshold_instance) {
                return i;
            }
        }
        return -1;
    };

    for (int i = 0; i < dim.d[2]; ++i) {
        for (int j = 0; j < dim.d[3]; ++j) {
            if ((int)mask.at<uchar>(i, j) == 0) {
                continue;
            }

            const cv::Vec2f& offset = offsets.at<cv::Vec2f>(i, j);
            cv::Point2f point(offset[1] + j, offset[0] + i);
            if (point.x > dim.d[3] || point.x < 0.f) continue;
            if (point.y > dim.d[2] || point.y < 0.f) continue;

            const cv::Vec4f& feature = features.at<cv::Vec4f>(i, j);
            int lane_index = findNearestFeature(feature);
            
            if (lane_index == -1) {
                laneLines.emplace_back(LaneLine({point}));
                laneFeatures.emplace_back(feature);
            } else {
                auto& laneline = laneLines[lane_index];
                auto& lanefeature = laneFeatures[lane_index];

                auto point_size = laneline.size(); 

                lanefeature = lanefeature.mul(cv::Vec4f::all(point_size)) + feature;
                lanefeature = lanefeature.mul(cv::Vec4f::all(1.f / (point_size + 1)));
                laneline.emplace_back(point);
            }
        }
    }

    for (auto itr = laneLines.begin(); itr != laneLines.end();) {
        if ((*itr).size() < 2) {
            itr = laneLines.erase(itr);
        } else {
            ++itr;
        }
    }

    return laneLines;
}

//!
//! \brief verify result
//!
//! \return whether output matches expectations
//!
bool PINetTensorrt::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    float *confidance, *offset, *instance;
    confidance = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[output_base_index + 0]));    
    offset     = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[output_base_index + 1]));    
    instance   = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[output_base_index + 2]));    
 
    nvinfer1::Dims confidanceDims = mOutputDims[output_base_index + 0];
    nvinfer1::Dims offsetDims     = mOutputDims[output_base_index + 1];
    nvinfer1::Dims instanceDims   = mOutputDims[output_base_index + 2];
    
    assert(confidanceDims.d[1] == 1);
    assert(offsetDims.d[1]     == 2);
    assert(instanceDims.d[1]   == 4);

    LaneLines lanelines = generateLaneLine(confidance, offset, instance);
    if (lanelines.empty())
        return false;

    cv::Scalar color[] = {{255,   0,   0}, {  0, 255,   0}, {  0,   0, 255}, 
                        {255, 255,   0}, {255,   0, 255}, {  0, 255, 255}, 
                        {255, 255, 255}, {100, 255,   0}, {100,   0, 255}, 
                        {255, 100,   0}, {  0, 100, 255}, {255,   0, 100}, 
                        {  0, 255, 100}};

    cv::Mat lanelineImage = mInputImage;
    for (int i = 0; i < lanelines.size(); ++i) {
        for (const auto& point : lanelines[i]) {
            cv::circle(lanelineImage, cv::Point2f(point * 8), 3, color[i], -1);
        }
    }

    if (sample::gLogger.getReportableSeverity() == sample::Logger::Severity::kINFO) {
        cv::imwrite("lanelines.jpg", lanelineImage);

        cv::imshow("lanelines", lanelineImage);
        cv::waitKey(0);
    }

    return true;
}
//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("./data/1492638000682869180");
    } 
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }

    char pwd[1024] = {0};
    getcwd(pwd, sizeof(pwd));

    params.onnxFileName = "pinet.onnx";
    params.inputTensorNames.push_back("input.1");
    //params.outputTensorNames.push_back("1431");
    params.outputTensorNames.push_back("input.672");
    params.outputTensorNames.push_back("1438");
    params.outputTensorNames.push_back("1445");
    //params.outputTensorNames.push_back("1679");
    params.outputTensorNames.push_back("input.1332");
    params.outputTensorNames.push_back("1686");
    params.outputTensorNames.push_back("1693");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./pinettensorrt [-h or --help] [-d or --datadir=<path to data path>] [--useDLACore=<int>]" << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data path, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)" << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    sample::setReportableSeverity(sample::Logger::Severity::kINFO);
    auto test = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(test);

    samplesCommon::OnnxSampleParams onnx_args = initializeSampleParams(args);
    PINetTensorrt sample(onnx_args);

    sample::gLogInfo << "Building and running a GPU inference engine for Onnx PINet" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(test);
    }

    std::vector<std::string> filenames;
    filenames.reserve(20480);
    for (size_t i = 0; i < onnx_args.dataDirs.size(); i++) {
        getFiles(onnx_args.dataDirs[i], ".jpg", filenames);
    }

    auto inference_begin_time = std::chrono::high_resolution_clock::now();

    for (const auto& filename : filenames) {
        sample.setImageFile(filename);
        if (!sample.infer()) {
            sample::gLogger.reportFail(test);
        }
    }

    auto inference_elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - inference_begin_time);

    sample::gLogger.reportPass(test);

    sample::gLogInfo << std::endl;

    sample::gLogInfo <<     "totally inference time      : " << inference_elapsed_time.count() / 1000.f << " milliseconds" << std::endl;
    if (filenames.size()) {
        sample::gLogInfo << "totally inference times     : " << filenames.size() << std::endl;
        sample::gLogInfo << "average inference time      : " << inference_elapsed_time.count() / filenames.size() / 1000.f << " milliseconds"<< std::endl;
    }

    if (total_inference_execute_times > 0) {
        sample::gLogInfo << "totally execute elapsed time: " << total_inference_execute_elasped_time / 1000.f << " milliseconds" << std::endl << std::endl;
        sample::gLogInfo << "inference execute times     : " << total_inference_execute_times << std::endl;
        sample::gLogInfo << "average execute elapsed time: " << total_inference_execute_elasped_time / total_inference_execute_times / 1000.f << " milliseconds" << std::endl << std::endl;
    }

    return 0;
}
