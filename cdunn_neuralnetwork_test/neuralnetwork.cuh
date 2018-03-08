#ifndef _NEURALNETWORK_H_
#define _NEURALNETWORK_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>

#include <stdio.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>

#include "readubyte.h"

#define BATCH_SIZE 64
#define BW 128
#define LEARNING_RATE 0.01
#define LR_GAMMA 0.0001
#define LR_POWER 0.75

using namespace std;

#define FatalError(s) {													\
    std::stringstream _where, _message;									\
    _where << __FILE__ << ':' << __LINE__;								\
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;	\
    std::cerr << _message.str() << "\nAborting...\n";					\
    cudaDeviceReset();													\
	getchar();															\
    exit(1);															\
}

#define checkCUDNN(status) {											\
    std::stringstream _error;											\
    if (status != CUDNN_STATUS_SUCCESS) {								\
      _error << "CUDNN failure: " << cudnnGetErrorString(status);		\
      FatalError(_error.str());											\
    }																	\
}


#define checkCudaErrors(status) {										\
    std::stringstream _error;											\
    if (status != 0) {													\
      _error << "Cuda failure: " << status;								\
      FatalError(_error.str());											\
    }																	\
}

class NeuralNetwork;
class Layer;
class ConvolutionLayer;
class MaxPoolLayer;
class FullyConnectedLayer;
class ActivationLayer;
class DataSet;
class OutputLayer;
class ResidualBlock;
class BranchLayer;

class Layer
{
public:
	friend class ConvolutionLayer;
	friend class MaxPoolLayer;
	friend class FullyConnectedLayer;
	friend class ActivationLayer;
	friend class DataSet;
	friend class OutputLayer;
	friend class ResidualBlock;
	friend class BranchLayer;
	friend class BatchNormLayer;

	inline int getOutputNumber() { return OutputNumber; }
	inline float *getData() { return device_data; }

	bool FromFile(const char *fileprefix);
	void ToFile(const char *fileprefix);
	virtual inline void ForwardPropagate() = 0;
	virtual inline void BackPropagate() = 0;
	virtual inline void Predict() { ForwardPropagate(); }
	virtual inline void UpdateWeights(float learning_rate) = 0;

	bool isFirstLayer = false;
	bool isSave = true;
protected:
	int InputNumber;			// 输入层神经元个数
	int OutputNumber;			// 输出层神经元个数
	int InputChannels;
	int OutputChannels;
	int KernelSize;
	int InputWidth;
	int InputHeight;
	int OutputWidth;
	int OutputHeight;
	int Padding;
	int Stride;
	vector<float> ParamW;		// 参数w
	vector<float> ParamB;		// 参数b

	float *device_data;
	float *device_param_w;
	float *device_param_b;
	float *device_grad_w;
	float *device_grad_b;
	float *device_diff_data;

	NeuralNetwork *neuralNetwork;
	Layer *LastLayer = nullptr;
	Layer *NextLayer = nullptr;
	cudnnTensorDescriptor_t TensorDesc;	

	

	virtual inline void deviceMalloc(int batchsize) = 0;
	virtual inline void deviceFree() = 0;
	virtual inline void CreateDescriptor(int batchsize) = 0;
	virtual inline void DestroyDescriptor() = 0;
};

/*
//	FullyConnectedLayer
//	全连接层类，包括所有全连接层相关的操作
*/
class FullyConnectedLayer : public Layer
{
public:
	FullyConnectedLayer(NeuralNetwork *neuralnetwork, Layer *lastlayer, int num);
	~FullyConnectedLayer();

	inline void ForwardPropagate();
	inline void BackPropagate();
	inline void UpdateWeights(float learning_rate);
private:
	inline void deviceMalloc(int batchsize);
	inline void deviceFree();
	inline void CreateDescriptor(int batchsize);
	inline void DestroyDescriptor();
};

/*
//	ActivationLayer
//	激活层类，包括所有激活层相关的操作
*/
class ActivationLayer : public Layer
{
public:
	friend class NeuralNetwork;

	ActivationLayer(NeuralNetwork *neuralnetwork, Layer *lastlayer, cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU, cudnnNanPropagation_t nanopt = CUDNN_PROPAGATE_NAN, double coef = 0.0);
	~ActivationLayer();

	inline void ForwardPropagate();
	inline void BackPropagate();
	inline void UpdateWeights(float learning_rate) {}
private:
	double Coef;
	cudnnActivationMode_t ActivationMode;
	cudnnNanPropagation_t NanOption;
	cudnnActivationDescriptor_t ActivationDesc;

	inline void deviceMalloc(int batchsize);
	inline void deviceFree();
	inline void CreateDescriptor(int batchsize);
	inline void DestroyDescriptor();
};

/*
//	ConvolutionLayer
//	卷积层类，包括所有卷积层相关的操作
*/
class ConvolutionLayer : public Layer
{
public:
	ConvolutionLayer(NeuralNetwork *neuralnetwork, Layer *lastlayer, int output_channels, int kernel_size, int padding = 0, int stride = 1);
	~ConvolutionLayer();

	inline void ForwardPropagate();
	inline void BackPropagate();
	inline void UpdateWeights(float learning_rate);
private:
	cudnnTensorDescriptor_t BiasTensorDesc;			// 张量描述符
	cudnnFilterDescriptor_t FilterDesc;				// 滤波器描述符
	cudnnConvolutionDescriptor_t ConvDesc;			// 卷积器描述符
	cudnnConvolutionFwdAlgo_t FwdAlgDesc;			// 前向传播算法描述符
	cudnnConvolutionBwdFilterAlgo_t BwdAlgDesc;		// 反向传播算法描述符
	cudnnConvolutionBwdDataAlgo_t BwdDataAlgDesc;	// 反向传播数据算法描述符

	inline void deviceMalloc(int batchsize);
	inline void deviceFree();
	inline void CreateDescriptor(int batchsize);
	inline void DestroyDescriptor();
};


/*
//	MaxPoolLayer
//	池化层类，包括所有池化层相关的操作
*/
class MaxPoolLayer : public Layer
{
public:
	MaxPoolLayer(NeuralNetwork *neuralnetwork, Layer *lastlayer, int size, int stride);
	~MaxPoolLayer();

	inline void ForwardPropagate();
	inline void BackPropagate();
	inline void UpdateWeights(float learning_rate) {}
private:
	cudnnPoolingDescriptor_t PoolDesc;		// 池化描述符

	inline void deviceMalloc(int batchsize);
	inline void deviceFree();
	inline void CreateDescriptor(int batchsize);
	inline void DestroyDescriptor();
};

/*
// DataSet
// 数据集相关类
*/
class DataSet : public Layer
{
public:
	DataSet();
	~DataSet();

	vector<uint8_t> TrainSet;
	vector<uint8_t> TrainLabels;
	vector<uint8_t> TestSet;
	vector<uint8_t> TestLabels;
	vector<float>	TrainSet_float;
	vector<float>	TrainLabels_float;

	inline float *getLabels() { return device_labels; }
	inline size_t getTrainSize() { return TrainSize; }
	inline size_t getTestSize() { return TestSize; }

	inline void ForwardPropagate() {}
	inline void BackPropagate() {}
	inline void UpdateWeights(float learning_rate) {}
private:
	float *device_labels;

	string TrainingSetName = "train-images.idx3-ubyte";
	string TrainingLabelsName = "train-labels.idx1-ubyte";
	string TestSetName = "t10k-images.idx3-ubyte";
	string TestLabelsName = "t10k-labels.idx1-ubyte";


	size_t TrainSize;
	size_t TestSize;

	inline void deviceMalloc(int batchsize);
	inline void deviceFree();
	inline void CreateDescriptor(int batchsize);
	inline void DestroyDescriptor();
};

class OutputLayer : public Layer
{
public:
	OutputLayer(NeuralNetwork *neuralnetwork, Layer *lastlayer);
	~OutputLayer();

	inline void ForwardPropagate();
	inline void BackPropagate();
	inline void UpdateWeights(float learning_rate) {}

private:
	float *device_loss_data;

	inline void deviceMalloc(int batchsize);
	inline void deviceFree();
	inline void CreateDescriptor(int batchsize);
	inline void DestroyDescriptor();
};

class BranchLayer : public Layer
{
public:
	BranchLayer(NeuralNetwork *neuralnetwork, Layer *lastlayer);
	~BranchLayer();

	ResidualBlock *ResBlock = nullptr;

	inline void ForwardPropagate();
	inline void BackPropagate();
	inline void UpdateWeights(float learning_rate);
private:
	inline void deviceMalloc(int batchsize);
	inline void deviceFree();
	inline void CreateDescriptor(int batchsize);
	inline void DestroyDescriptor();
};

// 要求输入和输出的宽度、高度和维度分别相同
class ResidualBlock : public Layer
{
public:
	ResidualBlock(NeuralNetwork *neuralnetwork, Layer *lastlayer);
	~ResidualBlock();

	inline void ForwardPropagate();
	inline void BackPropagate();
	inline void UpdateWeights(float learning_rate);
private:
	float *device_branch_data;

	inline void deviceMalloc(int batchsize);
	inline void deviceFree();
	inline void CreateDescriptor(int batchsize);
	inline void DestroyDescriptor();
};


class BatchNormLayer : Layer
{
public:
	BatchNormLayer(NeuralNetwork *neuralnetwork, Layer *branchlayer);
	~BatchNormLayer();

	inline void ForwardPropagate();
	inline void BackPropagate();
	inline void UpdateWeights(float learning_rate);
private:
	inline void deviceMalloc(int batchsize);
	inline void deviceFree();
	inline void CreateDescriptor(int batchsize);
	inline void DestroyDescriptor();
};


class NeuralNetwork
{
public:
	friend class Layer;
	friend class ConvolutionLayer;
	friend class MaxPoolLayer;
	friend class FullyConnectedLayer;
	friend class ActivationLayer;
	friend class DataSet;
	friend class OutputLayer;
	friend class ResidualBlock;
	friend class BranchLayer;
	friend class BatchNormLayer;

	NeuralNetwork();
	~NeuralNetwork();

	void Create();
	void Train(int iterations);
	void Test();
	void Destroy();

	void AddData(DataSet *dataset);
	void AddLayer(Layer *layer, bool isfirstlayer = false);
	
	DataSet				*Data;
	vector<Layer*>		Layers;

private:
	int GPUid = 0;
	cudnnHandle_t cudnnHandle;
	cublasHandle_t cublasHandle;
	float *device_ones;
	float *device_labels;
	size_t WorkspaceSize = 0;
	void *device_workspace = nullptr;

	void ForwardPropagate();
	void BackPropagate();
	void Predict();
	void UpdateWeights(float learning_rate);
};


#endif // !_NEURALNETWORK_H_

