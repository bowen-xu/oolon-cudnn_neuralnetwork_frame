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

class Layer
{
public:
	friend class NeuralNetwork;
	friend class ConvolutionLayer;
	friend class MaxPoolLayer;
	friend class FullyConnectedLayer;
	friend class ActivationLayer;
	friend class DataSet;
	friend class OutputLayer;
	int InputNumber;			// �������Ԫ����
	int OutputNumber;			// �������Ԫ����
	vector<float> ParamW;		// ����w
	vector<float> ParamB;		// ����b

protected:
	float *device_data;
	float *device_param_w;
	float *device_param_b;
	float *device_grad_w;
	float *device_grad_b;
	float *device_diff_data;

	Layer *LastLayer;
	cudnnHandle_t cudnnHandle;
	cudnnTensorDescriptor_t TensorDesc;	
	cudnnTensorDescriptor_t LastTensorDesc;

	virtual inline void deviceMalloc(int batchsize) = 0;
	virtual inline void deviceFree() = 0;
	virtual inline void CreateDescriptor(int batchsize) = 0;
	virtual inline void DestroyDescriptor() = 0;
};

/*
//	FullyConnectedLayer
//	ȫ���Ӳ��࣬��������ȫ���Ӳ���صĲ���
*/
class FullyConnectedLayer : public Layer
{
public:
	friend class NeuralNetwork;

	FullyConnectedLayer(Layer *lastlayer, int input_num, int output_num);
	~FullyConnectedLayer();
private:
	inline void deviceMalloc(int batchsize);
	inline void deviceFree();
	inline void CreateDescriptor(int batchsize);
	inline void DestroyDescriptor();
};

/*
//	ActivationLayer
//	������࣬�������м������صĲ���
*/
class ActivationLayer : public Layer
{
public:
	friend class NeuralNetwork;
	int Number;
	ActivationLayer(Layer *lastlayer, int num, cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU, cudnnNanPropagation_t nanopt = CUDNN_PROPAGATE_NAN, double coef = 0.0);
	~ActivationLayer();
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
//	������࣬�������о������صĲ���
*/
class ConvolutionLayer : public Layer
{
public:
	friend class NeuralNetwork;
	int InputChannels;
	int OutputChannels;
	int KernelSize;
	int InputWidth;
	int InputHeight;
	int OutputWidth;
	int OutputHeight;
	int Padding;
	int Stride;

	ConvolutionLayer(Layer *lastlayer, cudnnHandle_t cudnnhandle, int in_channels, int out_channels, int kernel_size, int in_width, int in_height, int padding = 0, int stride = 1);
	~ConvolutionLayer();

	inline void ForwardPropagate(void *device_workspace, size_t WorkspaceSize);
private:
	cudnnTensorDescriptor_t BiasTensorDesc;			// ����������
	cudnnFilterDescriptor_t FilterDesc;				// �˲���������
	cudnnConvolutionDescriptor_t ConvDesc;			// �����������
	cudnnConvolutionFwdAlgo_t FwdAlgDesc;			// ǰ�򴫲��㷨������
	cudnnConvolutionBwdFilterAlgo_t BwdAlgDesc;		// ���򴫲��㷨������
	cudnnConvolutionBwdDataAlgo_t BwdDataAlgDesc;	// ���򴫲������㷨������

	size_t WorkspaceSize = 0;

	inline void deviceMalloc(int batchsize);
	inline void deviceFree();
	inline void CreateDescriptor(int batchsize);
	inline void DestroyDescriptor();
};




/*
//	MaxPoolLayer
//	�ػ����࣬�������гػ�����صĲ���
*/
class MaxPoolLayer : public Layer
{
public:
	friend class NeuralNetwork;
	int OutputWidth;
	int OutputHeight;
	int OutputChannels;
	int Size;
	int Stride;
	
	MaxPoolLayer(Layer *lastlayer, cudnnHandle_t cudnnhandle, int size, int stride, ConvolutionLayer &lastConv);
	~MaxPoolLayer();

	inline void ForwardPropagate();
private:
	cudnnPoolingDescriptor_t PoolDesc;		// �ػ�������

	inline void deviceMalloc(int batchsize);
	inline void deviceFree();
	inline void CreateDescriptor(int batchsize);
	inline void DestroyDescriptor();
};

class DataSet : public Layer
{
public:
	friend class NeuralNetwork;
	DataSet();
	~DataSet();

	size_t Width;
	size_t Height;
	size_t Channels = 1;
	

	vector<uint8_t> TrainSet;
	vector<uint8_t> TrainLabels;
	vector<uint8_t> TestSet;
	vector<uint8_t> TestLabels;
	vector<float>	TrainSet_float;
	vector<float>	TrainLabels_float;

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
	friend class NeuralNetwork;

	OutputLayer(Layer *lastlayer, int num);
	~OutputLayer();

	size_t Number;

private:
	float *device_loss_data;

	inline void deviceMalloc(int batchsize);
	inline void deviceFree();
	inline void CreateDescriptor(int batchsize);
	inline void DestroyDescriptor();
};

class NeuralNetwork
{
public:
	NeuralNetwork();
	void Create();
	void Train(int iterations);
	void Test();
	void Destroy();
private:
	int GPUid = 0;
	cudnnHandle_t cudnnHandle;
	cublasHandle_t cublasHandle;
	float *device_ones;
	size_t WorkspaceSize = 0;
	void *device_workspace = nullptr;

	DataSet				*Image;
	ConvolutionLayer	*Conv1;
	MaxPoolLayer		*Pool1;
	ConvolutionLayer	*Conv2;
	MaxPoolLayer		*Pool2;
	FullyConnectedLayer	*FC1;
	ActivationLayer		*ACTN1;
	FullyConnectedLayer	*FC2;
	OutputLayer			*RSLT;


	void ForwardPropagate();
	void BackPropagate();
	void UpdateWeights(float learning_rate);
};


#endif // !_NEURALNETWORK_H_

