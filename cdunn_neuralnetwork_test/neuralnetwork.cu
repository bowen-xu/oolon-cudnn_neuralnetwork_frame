#include "neuralnetwork.cuh"

/**
* Computes ceil(x / y) for integral nonnegative values.
*/
static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
	return (nominator + denominator - 1) / denominator;
}

/**
* Fills a floating-point array with ones.
*
* @param vec The array to fill.
* @param size The number of elements in the array.
*/
__global__ void FillOnes(float *vec, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	vec[idx] = 1.0f;
}
/**
* Computes the backpropagation results of the Softmax loss for each result in a batch.
* Uses the softmax values obtained from forward propagation to compute the difference.
*
* @param label The training batch label values.
* @param num_labels The number of possible labels.
* @param batch_size The size of the trained batch.
* @param diff The resulting gradient.
*/
__global__ void SoftmaxLossBackprop(const float *label, const float *data, float *diff, int num_labels, int batch_size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batch_size)
		return;

	const int label_value = static_cast<int>(label[idx]);

	// For each item in the batch, decrease the result of the label's value by 1
	diff[idx * num_labels + label_value] = data[idx * num_labels + label_value] - 1.0f;
}



/*
// FullyConnectedLayer
*/
FullyConnectedLayer::FullyConnectedLayer(NeuralNetwork *neuralnetwork, Layer *lastlayer, int num)
{
	InputNumber = lastlayer->OutputNumber;
	OutputNumber = num;
	InputChannels = OutputChannels = 1;
	Padding = 0;
	Stride = 1;
	KernelSize = 1;

	ParamW.resize(InputNumber * OutputNumber);
	ParamB.resize(OutputNumber);

	neuralNetwork = neuralnetwork;
	LastLayer = lastlayer;
	lastlayer->NextLayer = this;

	//random_device rd;
	//mt19937 gen(rd());
	//float wfc = sqrt(3.0f / (InputNumber * OutputNumber));
	//std::uniform_real_distribution<> dfc(-wfc, wfc);
	//for (auto&& iter : ParamW)
	//	iter = static_cast<float>(dfc(gen));
	//for (auto&& iter : ParamB)
	//	iter = static_cast<float>(dfc(gen));

	static int index = 0;
	index++;
	char str[256];
	sprintf(str, "FC%d", index);

	if (isSave)
	{
		random_device rd;
		mt19937 gen(rd());
		float wfc = sqrt(3.0f / (InputNumber * OutputNumber));
		std::uniform_real_distribution<> dfc(-wfc, wfc);
		for (auto&& iter : ParamW)
			iter = static_cast<float>(dfc(gen));
		for (auto&& iter : ParamB)
			iter = static_cast<float>(dfc(gen));

		ToFile(str);
	}
	else
	{
		FromFile(str);
	}

	CreateDescriptor(BATCH_SIZE);
	deviceMalloc(BATCH_SIZE);
}


FullyConnectedLayer::~FullyConnectedLayer()
{
	DestroyDescriptor();
	deviceFree();
}

inline void FullyConnectedLayer::ForwardPropagate()
{
	static float alpha = 1.0f, beta = 0.0f;
	// Forward propagate neurons using weights (fc1 = pfc1'*pool2)
	checkCudaErrors(cublasSgemm(neuralNetwork->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
		OutputNumber, BATCH_SIZE, InputNumber, &alpha, device_param_w, InputNumber,
		LastLayer->device_data, InputNumber, &beta, device_data, OutputNumber));
	// Add bias using GEMM's "beta" (fc1 += pfc1bias*1_vec')
	checkCudaErrors(cublasSgemm(neuralNetwork->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		OutputNumber, BATCH_SIZE, 1, &alpha, device_param_b, OutputNumber, neuralNetwork->device_ones, 1, &alpha,
		device_data, OutputNumber));
}

inline void FullyConnectedLayer::BackPropagate()
{
	static float alpha = 1.0f, beta = 0.0f;
	
	checkCudaErrors(cublasSgemm(neuralNetwork->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, InputNumber, OutputNumber, BATCH_SIZE,
		&alpha, LastLayer->device_data, InputNumber, NextLayer->device_diff_data, OutputNumber, &beta, device_grad_w, InputNumber));
	
	checkCudaErrors(cublasSgemv(neuralNetwork->cublasHandle, CUBLAS_OP_N, OutputNumber, BATCH_SIZE,
		&alpha, NextLayer->device_diff_data, OutputNumber, neuralNetwork->device_ones, 1, &beta, device_grad_b, 1));
	// Compute derivative with respect to data (for previous layer): pfc2*dfc2smax (500x10*10xN)
	if (!isFirstLayer)
	{
		checkCudaErrors(cublasSgemm(neuralNetwork->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, InputNumber, BATCH_SIZE, OutputNumber,
			&alpha, device_param_w, InputNumber, NextLayer->device_diff_data, OutputNumber, &beta, device_diff_data, InputNumber));// ���һ��device_diff_data ��Ϊdevice_diff_data
	}
}

inline void FullyConnectedLayer::UpdateWeights(float learning_rate)
{
	float alpha = -learning_rate;

	checkCudaErrors(cublasSaxpy(neuralNetwork->cublasHandle, static_cast<int>(ParamW.size()),
		&alpha, device_grad_w, 1, device_param_w, 1));
	checkCudaErrors(cublasSaxpy(neuralNetwork->cublasHandle, static_cast<int>(ParamB.size()),
		&alpha, device_grad_b, 1, device_param_b, 1));
}

inline void FullyConnectedLayer::deviceMalloc(int batchsize)
{
	// ǰ�򴫲�����
	checkCudaErrors(cudaMalloc(&device_data, sizeof(float) * batchsize * OutputNumber));	// GPU�и����ݿ��ٿռ�
																							// ����
	checkCudaErrors(cudaMalloc(&device_param_w, sizeof(float) * ParamW.size()));			// GPU�и�����w���ٿռ�
	checkCudaErrors(cudaMalloc(&device_param_b, sizeof(float) * ParamB.size()));			// GPU�и�����b���ٿռ�
																							// �ݶ�
	checkCudaErrors(cudaMalloc(&device_grad_w, sizeof(float) * ParamW.size()));				// GPU�и��ݶ�w���ٿռ�
	checkCudaErrors(cudaMalloc(&device_grad_b, sizeof(float) * ParamB.size()));				// GPU�и��ݶ�b���ٿռ�
																							// ���򴫲�����
	checkCudaErrors(cudaMalloc(&device_diff_data, sizeof(float) * batchsize * InputNumber));

	// ������ʼ�����ݵ�GPU
	checkCudaErrors(cudaMemcpyAsync(device_param_w, &ParamW[0], sizeof(float) * ParamW.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(device_param_b, &ParamB[0], sizeof(float) * ParamB.size(), cudaMemcpyHostToDevice));
}

inline void FullyConnectedLayer::deviceFree()
{
	checkCudaErrors(cudaFree(device_data));
	checkCudaErrors(cudaFree(device_param_w));
	checkCudaErrors(cudaFree(device_param_b));
	checkCudaErrors(cudaFree(device_grad_w));
	checkCudaErrors(cudaFree(device_grad_b));
	checkCudaErrors(cudaFree(device_diff_data));
	checkCudaErrors(cudaFree(device_param_w));
	checkCudaErrors(cudaFree(device_param_b));
}

inline void FullyConnectedLayer::CreateDescriptor(int batchsize)
{
	checkCUDNN(cudnnCreateTensorDescriptor(&TensorDesc));

	//
	checkCUDNN(cudnnSetTensor4dDescriptor(TensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchsize, OutputNumber, 1, 1));
}

inline void FullyConnectedLayer::DestroyDescriptor()
{
	checkCUDNN(cudnnDestroyTensorDescriptor(TensorDesc));
}


/*
// ActivationLayer
*/
ActivationLayer::ActivationLayer(NeuralNetwork *neuralnetwork, Layer *lastlayer, cudnnActivationMode_t mode, cudnnNanPropagation_t nanopt, double coef)
{
	InputNumber = OutputNumber = lastlayer->OutputNumber;
	InputHeight = OutputHeight = lastlayer->OutputHeight;
	InputWidth = OutputWidth = lastlayer->OutputWidth;
	InputChannels = OutputChannels = lastlayer->OutputChannels;
	Padding = 0;
	KernelSize = 1;
	Stride = 1;

	ActivationMode = mode;
	NanOption = nanopt;
	Coef = coef;

	neuralNetwork = neuralnetwork;
	LastLayer = lastlayer;
	lastlayer->NextLayer = this;

	TensorDesc = LastLayer->TensorDesc;

	CreateDescriptor(BATCH_SIZE);
	deviceMalloc(BATCH_SIZE);
}

ActivationLayer::~ActivationLayer()
{
	DestroyDescriptor();
	deviceFree();
}

inline void ActivationLayer::ForwardPropagate()
{
	static float alpha = 1.0f, beta = 0.0f;
	checkCUDNN(cudnnActivationForward(neuralNetwork->cudnnHandle, ActivationDesc, &alpha,
		LastLayer->TensorDesc, LastLayer->device_data, &beta, LastLayer->TensorDesc, device_data));
}

inline void ActivationLayer::BackPropagate()
{
	static float alpha = 1.0f, beta = 0.0f;
	if (!isFirstLayer)
	{
		checkCUDNN(cudnnActivationBackward(neuralNetwork->cudnnHandle, ActivationDesc, &alpha,
			LastLayer->TensorDesc, device_data, LastLayer->TensorDesc, NextLayer->device_diff_data,
			LastLayer->TensorDesc, LastLayer->device_data, &beta, LastLayer->TensorDesc, device_diff_data));
	}
}

inline void ActivationLayer::deviceMalloc(int batchsize)
{
	// ǰ�򴫲�����
	checkCudaErrors(cudaMalloc(&device_data, sizeof(float) * batchsize * OutputNumber));
	// ���򴫲�����
	checkCudaErrors(cudaMalloc(&device_diff_data, sizeof(float) * batchsize * InputNumber));
}

inline void ActivationLayer::deviceFree()
{
	checkCudaErrors(cudaFree(device_data));
	checkCudaErrors(cudaFree(device_diff_data));
}

inline void ActivationLayer::CreateDescriptor(int batchsize)
{
	// ����������
	checkCUDNN(cudnnCreateActivationDescriptor(&ActivationDesc));

	// ����������
	checkCUDNN(cudnnSetActivationDescriptor(ActivationDesc, ActivationMode, NanOption, Coef));
}

inline void ActivationLayer::DestroyDescriptor()
{
	checkCUDNN(cudnnDestroyActivationDescriptor(ActivationDesc));
}



/*
// ConvolutionLayer
*/
ConvolutionLayer::ConvolutionLayer(NeuralNetwork *neuralnetwork, Layer *lastlayer, int output_channels, int kernel_size, int padding, int stride)
{
	InputWidth = lastlayer->OutputWidth;
	InputHeight = lastlayer->OutputHeight;
	OutputHeight = (InputHeight + 2 * padding - kernel_size) / stride + 1;
	OutputWidth = (InputWidth + 2 * padding - kernel_size) / stride + 1;
	InputChannels = lastlayer->OutputChannels;
	OutputChannels = output_channels;
	InputNumber = InputHeight * InputWidth * InputChannels;
	OutputNumber = OutputHeight * OutputWidth * OutputChannels;
	KernelSize = kernel_size;
	Padding = padding;
	Stride = stride;
	
	ParamW.resize(InputChannels * KernelSize * KernelSize * OutputChannels);
	ParamB.resize(OutputChannels);

	neuralNetwork = neuralnetwork;
	LastLayer = lastlayer;
	lastlayer->NextLayer = this;

	static int index = 0;
	index++;
	char str[256];
	sprintf(str, "Conv%d", index);

	if (isSave)
	{
		random_device rd;
		mt19937 gen(rd());
		float wconv = sqrt(3.0f / (KernelSize * KernelSize * InputChannels));
		std::uniform_real_distribution<> dconv(-wconv, wconv);
		for (auto&& iter : ParamW)
			iter = static_cast<float>(dconv(gen));
		for (auto&& iter : ParamB)
			iter = static_cast<float>(dconv(gen));

		ToFile(str);
	}
	else
	{

		FromFile(str);
	}

	CreateDescriptor(BATCH_SIZE);
	deviceMalloc(BATCH_SIZE);

}

ConvolutionLayer::~ConvolutionLayer()
{
	DestroyDescriptor();
	deviceFree();
}


inline void ConvolutionLayer::ForwardPropagate()
{
	static float alpha = 1.0f, beta = 0.0f;
	checkCUDNN(cudnnConvolutionForward(
		neuralNetwork->cudnnHandle, &alpha, LastLayer->TensorDesc,
		LastLayer->device_data, FilterDesc, device_param_w, ConvDesc,
		FwdAlgDesc, neuralNetwork->device_workspace, neuralNetwork->WorkspaceSize, &beta,
		TensorDesc, device_data));

	checkCUDNN(cudnnAddTensor(neuralNetwork->cudnnHandle, &alpha, BiasTensorDesc,
		device_param_b, &alpha, TensorDesc, device_data));
}

inline void ConvolutionLayer::BackPropagate()
{
	static float alpha = 1.0f, beta = 0.0f;

	checkCUDNN(cudnnConvolutionBackwardBias(neuralNetwork->cudnnHandle, &alpha, TensorDesc,
		NextLayer->device_diff_data, &beta, BiasTensorDesc, device_grad_b));


	checkCUDNN(cudnnConvolutionBackwardFilter(neuralNetwork->cudnnHandle, &alpha, LastLayer->TensorDesc,
		LastLayer->device_data, TensorDesc, NextLayer->device_diff_data, ConvDesc,
		BwdAlgDesc, neuralNetwork->device_workspace, neuralNetwork->WorkspaceSize,
		&beta, FilterDesc, device_grad_w));

	if (!isFirstLayer)
	{
		checkCUDNN(cudnnConvolutionBackwardData(neuralNetwork->cudnnHandle, &alpha, FilterDesc,
			device_param_w, TensorDesc, NextLayer->device_diff_data, ConvDesc,
			BwdDataAlgDesc, neuralNetwork->device_workspace, neuralNetwork->WorkspaceSize,
			&beta, LastLayer->TensorDesc, device_diff_data));
	}	
}

inline void ConvolutionLayer::UpdateWeights(float learning_rate)
{
	float alpha = -learning_rate;

	checkCudaErrors(cublasSaxpy(neuralNetwork->cublasHandle, static_cast<int>(ParamW.size()),
		&alpha, device_grad_w, 1, device_param_w, 1));
	checkCudaErrors(cublasSaxpy(neuralNetwork->cublasHandle, static_cast<int>(ParamB.size()),
		&alpha, device_grad_b, 1, device_param_b, 1));
}

inline void ConvolutionLayer::deviceMalloc(int batchsize)
{
	// ǰ�򴫲�����
	checkCudaErrors(cudaMalloc(&device_data, sizeof(float) * batchsize * OutputChannels * OutputHeight * OutputWidth));

	// ����
	checkCudaErrors(cudaMalloc(&device_param_w, sizeof(float) * ParamW.size()));
	checkCudaErrors(cudaMalloc(&device_param_b, sizeof(float) * ParamB.size()));
	// �ݶ�
	checkCudaErrors(cudaMalloc(&device_grad_w, sizeof(float) * ParamW.size()));
	checkCudaErrors(cudaMalloc(&device_grad_b, sizeof(float) * ParamB.size()));
	// ���򴫲�����
	checkCudaErrors(cudaMalloc(&device_diff_data, sizeof(float) * batchsize * InputNumber));

	// ������ʼ�����ݵ�GPU
	checkCudaErrors(cudaMemcpyAsync(device_param_w, &ParamW[0], sizeof(float) * ParamW.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(device_param_b, &ParamB[0], sizeof(float) * ParamB.size(), cudaMemcpyHostToDevice));
}

inline void ConvolutionLayer::deviceFree()
{
	checkCudaErrors(cudaFree(device_data));
	checkCudaErrors(cudaFree(device_param_w));
	checkCudaErrors(cudaFree(device_param_b));
	checkCudaErrors(cudaFree(device_grad_w));
	checkCudaErrors(cudaFree(device_grad_b));
	checkCudaErrors(cudaFree(device_diff_data));
	checkCudaErrors(cudaFree(device_param_w));
	checkCudaErrors(cudaFree(device_param_b));

}

inline void ConvolutionLayer::CreateDescriptor(int batchsize)
{
	size_t tempsize;
	// ��������
	checkCUDNN(cudnnCreateTensorDescriptor(&TensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&BiasTensorDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&FilterDesc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&ConvDesc));

	// ��������
	checkCUDNN(cudnnSetTensor4dDescriptor(BiasTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, OutputChannels, 1, 1));
	checkCUDNN(cudnnSetFilter4dDescriptor(FilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OutputChannels, InputChannels, KernelSize, KernelSize));
	checkCUDNN(cudnnSetConvolution2dDescriptor(ConvDesc, Padding, Padding, Stride, Stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
	checkCUDNN(cudnnSetTensor4dDescriptor(TensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchsize, OutputChannels, OutputHeight, OutputWidth));

	// ǰ�򴫲��㷨����
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(neuralNetwork->cudnnHandle, LastLayer->TensorDesc, FilterDesc, ConvDesc, TensorDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &FwdAlgDesc));
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(neuralNetwork->cudnnHandle, LastLayer->TensorDesc, FilterDesc, ConvDesc, TensorDesc, FwdAlgDesc, &tempsize));
	neuralNetwork->WorkspaceSize = max(neuralNetwork->WorkspaceSize, tempsize);

	// ���򴫲��㷨����
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(neuralNetwork->cudnnHandle, LastLayer->TensorDesc, TensorDesc, ConvDesc, FilterDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &BwdAlgDesc));
	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(neuralNetwork->cudnnHandle, LastLayer->TensorDesc, TensorDesc, ConvDesc, FilterDesc, BwdAlgDesc, &tempsize));
	neuralNetwork->WorkspaceSize = max(neuralNetwork->WorkspaceSize, tempsize);
	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(neuralNetwork->cudnnHandle, FilterDesc, TensorDesc, ConvDesc, LastLayer->TensorDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &BwdDataAlgDesc));
	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(neuralNetwork->cudnnHandle, FilterDesc, TensorDesc, ConvDesc, LastLayer->TensorDesc, BwdDataAlgDesc, &tempsize));
	neuralNetwork->WorkspaceSize = max(neuralNetwork->WorkspaceSize, tempsize);
}

inline void ConvolutionLayer::DestroyDescriptor()
{
	checkCUDNN(cudnnDestroyTensorDescriptor(TensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(BiasTensorDesc));
	checkCUDNN(cudnnDestroyFilterDescriptor(FilterDesc));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(ConvDesc));
}

/*
// MaxPoolLayer
*/
MaxPoolLayer::MaxPoolLayer(NeuralNetwork *neuralnetwork, Layer *lastlayer, int size, int stride)
{
	InputChannels = OutputChannels = lastlayer->OutputChannels;
	InputWidth = lastlayer->OutputWidth;
	InputHeight = lastlayer->OutputHeight;
	OutputWidth = InputWidth / stride;
	OutputHeight = InputHeight / stride;
	InputNumber = InputWidth * InputHeight * InputChannels;
	OutputNumber = OutputWidth * OutputHeight * OutputChannels;
	KernelSize = size;
	Stride = stride;
	Padding = 0;

	neuralNetwork = neuralnetwork;
	LastLayer = lastlayer;
	lastlayer->NextLayer = this;

	CreateDescriptor(BATCH_SIZE);
	deviceMalloc(BATCH_SIZE);
}

MaxPoolLayer::~MaxPoolLayer()
{
	DestroyDescriptor();
	deviceFree();
}

inline void MaxPoolLayer::ForwardPropagate()
{
	static float alpha = 1.0f, beta = 0.0f;
	checkCUDNN(cudnnPoolingForward(neuralNetwork->cudnnHandle, PoolDesc, &alpha, LastLayer->TensorDesc,
		LastLayer->device_data, &beta, TensorDesc, device_data));
}

inline void MaxPoolLayer::BackPropagate()
{
	static float alpha = 1.0f, beta = 0.0f;
	if (!isFirstLayer)
	{
		checkCUDNN(cudnnPoolingBackward(neuralNetwork->cudnnHandle, PoolDesc, &alpha,
			TensorDesc, device_data, TensorDesc, NextLayer->device_diff_data,
			LastLayer->TensorDesc, LastLayer->device_data, &beta, LastLayer->TensorDesc, device_diff_data));
	}
	
}

inline void MaxPoolLayer::deviceMalloc(int batchsize)
{
	// ǰ�򴫲�����
	checkCudaErrors(cudaMalloc(&device_data, sizeof(float) * batchsize * OutputNumber));
	// ���򴫲�����
	checkCudaErrors(cudaMalloc(&device_diff_data, sizeof(float) * batchsize * InputNumber));
}

inline void MaxPoolLayer::deviceFree()
{
	checkCudaErrors(cudaFree(device_data));
	checkCudaErrors(cudaFree(device_diff_data));
}

inline void MaxPoolLayer::CreateDescriptor(int batchsize)
{
	// ����������
	checkCUDNN(cudnnCreateTensorDescriptor(&TensorDesc));
	checkCUDNN(cudnnCreatePoolingDescriptor(&PoolDesc));

	// ����������
	checkCUDNN(cudnnSetPooling2dDescriptor(PoolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, KernelSize, KernelSize, 0, 0, Stride, Stride));
	checkCUDNN(cudnnSetTensor4dDescriptor(TensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchsize, OutputChannels, OutputHeight, OutputWidth));
}

inline void MaxPoolLayer::DestroyDescriptor()
{
	checkCUDNN(cudnnDestroyTensorDescriptor(TensorDesc));
	checkCUDNN(cudnnDestroyPoolingDescriptor(PoolDesc));
}


/*
// DataSet
*/
DataSet::DataSet()
{
	size_t width, height;
	printf("Reading input data\n");
	
	size_t train_size = ReadUByteDataset(TrainingSetName.c_str(), TrainingLabelsName.c_str(), nullptr, nullptr, width, height);
	size_t test_size = ReadUByteDataset(TestSetName.c_str(), TestLabelsName.c_str(), nullptr, nullptr, width, height);
	TrainSize = train_size;
	TestSize = test_size;
	if (train_size == 0)
		exit(1);

	InputChannels = OutputChannels = 1;
	InputWidth = OutputWidth = width;
	InputHeight = OutputHeight = height;
	InputNumber = InputHeight * InputWidth * InputChannels;
	OutputNumber = OutputHeight * OutputWidth * OutputChannels;
	Padding = 0;
	KernelSize = 1;
	Stride = 1;

	TrainSet.resize(train_size * OutputNumber);
	TrainLabels.resize(train_size);
	TestSet.resize(test_size * OutputNumber);
	TestLabels.resize(test_size);

	if (ReadUByteDataset(TrainingSetName.c_str(), TrainingLabelsName.c_str(), &TrainSet[0], &TrainLabels[0], width, height) != train_size)
		exit(2);
	if (ReadUByteDataset(TestSetName.c_str(), TestLabelsName.c_str(), &TestSet[0], &TestLabels[0], width, height) != test_size)
		exit(3);

	printf("Done. Training dataset size: %d, Test dataset size: %d\n", (int)train_size, (int)test_size);
	
	// Normalize training set to be in [0,1]
	printf("Normalizing training data...\n");
	TrainSet_float.resize(TrainSet.size());
	TrainLabels_float.resize(TrainLabels.size());
	for (size_t i = 0; i < train_size * OutputNumber; ++i)
		TrainSet_float[i] = (float)TrainSet[i] / 255.0f;

	for (size_t i = 0; i < train_size; ++i)
		TrainLabels_float[i] = (float)TrainLabels[i];

	CreateDescriptor(BATCH_SIZE);
	deviceMalloc(BATCH_SIZE);
}

DataSet::~DataSet()
{
	DestroyDescriptor();
	deviceFree();
}

//inline void DataSet::ForwardPropagate()
//{
//	static int iter = 0;
//	iter++;
//	int imageid = iter % (TrainSize / BATCH_SIZE);
//	/*checkCudaErrors(cudaMemcpyAsync(device_data, &((TrainSet_float)[imageid * BATCH_SIZE * OutputNumber]),
//		sizeof(float) * BATCH_SIZE * OutputNumber, cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpyAsync(device_labels, &((TrainLabels_float)[imageid * BATCH_SIZE]),
//		sizeof(float) * BATCH_SIZE, cudaMemcpyHostToDevice));*/
//	checkCudaErrors(cudaMemcpyAsync(device_data, &((TrainSet_float)[imageid * BATCH_SIZE * OutputNumber]),
//		sizeof(float) * BATCH_SIZE * OutputNumber, cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpyAsync(device_labels, &((TrainLabels_float)[imageid * BATCH_SIZE]),
//		sizeof(float) * BATCH_SIZE, cudaMemcpyHostToDevice));
//}

inline void DataSet::deviceMalloc(int batchsize)
{
	// ǰ�򴫲�����
	checkCudaErrors(cudaMalloc(&device_data, sizeof(float) * batchsize * OutputNumber));
	checkCudaErrors(cudaMalloc(&device_labels, sizeof(float) * batchsize));
}

inline void DataSet::deviceFree()
{
	checkCudaErrors(cudaFree(device_data));
	checkCudaErrors(cudaFree(device_labels));
}

inline void DataSet::CreateDescriptor(int batchsize)
{
	// ��������
	checkCUDNN(cudnnCreateTensorDescriptor(&TensorDesc)); //

	// ��������
	checkCUDNN(cudnnSetTensor4dDescriptor(TensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchsize, OutputChannels, OutputHeight, OutputWidth));
}

inline void DataSet::DestroyDescriptor()
{
	checkCUDNN(cudnnDestroyTensorDescriptor(TensorDesc));
}

/*
// OutputLayer
*/
OutputLayer::OutputLayer(NeuralNetwork *neuralnetwork, Layer *lastlayer)
{
	OutputNumber = InputNumber = lastlayer->OutputNumber;
	OutputHeight = InputHeight = lastlayer->OutputHeight;
	OutputWidth = InputWidth = lastlayer->OutputWidth;
	OutputChannels = InputChannels = lastlayer->OutputChannels;
	Padding = 0;
	KernelSize = 1;
	Stride = 1;

	neuralNetwork = neuralnetwork;
	LastLayer = lastlayer;
	lastlayer->NextLayer = this;

	CreateDescriptor(BATCH_SIZE);
	deviceMalloc(BATCH_SIZE);
}

OutputLayer::~OutputLayer()
{
	DestroyDescriptor();
	deviceFree();
}

inline void OutputLayer::ForwardPropagate()
{
	static float alpha = 1.0f, beta = 0.0f;
	checkCUDNN(cudnnSoftmaxForward(neuralNetwork->cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
		&alpha, LastLayer->TensorDesc, LastLayer->device_data, &beta, LastLayer->TensorDesc, device_data));
}

inline void OutputLayer::BackPropagate()
{
	static float scalVal = 1.0f / static_cast<float>(BATCH_SIZE);
	static float alpha = 1.0f, beta = 0.0f;
	// Initialization (using the training error function)
	//checkCudaErrors(cudaMemcpyAsync(device_diff_data, device_data, sizeof(float) * BATCH_SIZE * LastLayer->OutputNumber, cudaMemcpyDeviceToDevice));

	// Softmax layer
	SoftmaxLossBackprop <<<RoundUp(BATCH_SIZE, BW), BW >>> (neuralNetwork->device_labels, device_data, device_diff_data, LastLayer->OutputNumber, BATCH_SIZE);

	// Accounting for batch size in SGD
	//checkCudaErrors(cublasSscal(neuralNetwork->cublasHandle, LastLayer->OutputNumber * BATCH_SIZE, &scalVal, device_diff_data, 1));
	checkCudaErrors(cudnnSoftmaxBackward(neuralNetwork->cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
		&alpha, LastLayer->TensorDesc, device_data, LastLayer->TensorDesc, device_diff_data, &beta, LastLayer->TensorDesc, device_diff_data));// ���һ��device_diff_data ��Ϊdevice_diff_data
}

inline void OutputLayer::deviceMalloc(int batchsize)
{
	// ǰ�򴫲�����
	checkCudaErrors(cudaMalloc(&device_data, sizeof(float) * batchsize * OutputNumber));
	// ���򴫲�����
	checkCudaErrors(cudaMalloc(&device_diff_data, sizeof(float) * batchsize * InputNumber));
	checkCudaErrors(cudaMalloc(&device_loss_data, sizeof(float) * batchsize * OutputNumber));
}

inline void OutputLayer::deviceFree()
{
	checkCudaErrors(cudaFree(device_data));
	checkCudaErrors(cudaFree(device_diff_data));
	checkCudaErrors(cudaFree(device_loss_data));
}

inline void OutputLayer::CreateDescriptor(int batchsize)
{

}

inline void OutputLayer::DestroyDescriptor()
{

}


/*
// NeuralNetwork
*/
NeuralNetwork::NeuralNetwork()
{
	int gpu_num;
	checkCudaErrors(cudaGetDeviceCount(&gpu_num));
	if (GPUid < 0 || GPUid >= gpu_num)
	{
		printf("ERROR: Invalid GPU ID %d (There are %d GPUs on this machine)\n", GPUid, gpu_num);
		exit(4);
	}

	checkCudaErrors(cublasCreate(&cublasHandle));
	checkCUDNN(cudnnCreate(&cudnnHandle));
}

NeuralNetwork::~NeuralNetwork()
{
	
}

void NeuralNetwork::AddData(DataSet *dataset)
{
	Data = dataset;
}

void NeuralNetwork::AddLayer(Layer *layer, bool isfirstlayer)
{
	if (isfirstlayer)
	{
		layer->isFirstLayer = true;
	}
	Layers.push_back(layer);
}

void NeuralNetwork::Create()
{
	/*Image = new DataSet();
	Conv1 = new ConvolutionLayer(this, Image, 20, 5);
	Pool1 = new MaxPoolLayer(this, Conv1, 2, 2);
	Conv2 = new ConvolutionLayer(this, Pool1, 50, 5);
	Pool2 = new MaxPoolLayer(this, Conv2, 2, 2);
	FC1 = new FullyConnectedLayer(this, Pool2, 500);
	ACTN1 = new ActivationLayer(this, FC1);
	FC2 = new FullyConnectedLayer(this, ACTN1, 10);
	RSLT = new OutputLayer(this, FC2);*/
	
	device_labels = Data->getLabels();
	checkCudaErrors(cudaMalloc(&device_ones, sizeof(float)* BATCH_SIZE));
	FillOnes <<<RoundUp(BATCH_SIZE, BW), BW>>> (device_ones, BATCH_SIZE);
	if (WorkspaceSize > 0)
		checkCudaErrors(cudaMalloc(&device_workspace, WorkspaceSize));

}

void NeuralNetwork::Destroy()
{
	delete Data;
	while (!Layers.empty())
	{
		delete Layers.back();
		Layers.pop_back();
	}
	//delete Image;
	//delete Conv1;
	//delete Pool1;
	//delete Conv2;
	//delete Pool2;
	//delete FC1;
	//delete ACTN1;
	//delete FC2;
	//delete RSLT;

	checkCudaErrors(cudaFree(device_ones));
	if (device_workspace != nullptr)
		checkCudaErrors(cudaFree(device_workspace));
	checkCudaErrors(cudaDeviceReset());
}

void NeuralNetwork::Train(int iterations)
{
	printf("Training...\n");
	checkCudaErrors(cudaDeviceSynchronize());
	auto t1 = std::chrono::high_resolution_clock::now();

	size_t train_size = Data->getTrainSize();
	float *device_data = Data->getData();

	for (int iter = 0; iter < iterations; ++iter)
	{
		int imageid = iter % (train_size / BATCH_SIZE);
		checkCudaErrors(cudaMemcpyAsync(device_data, &((Data->TrainSet_float)[imageid * BATCH_SIZE * Data->getOutputNumber()]),
			sizeof(float) * BATCH_SIZE * Data->getOutputNumber(), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyAsync(device_labels, &((Data->TrainLabels_float)[imageid * BATCH_SIZE]),
			sizeof(float) * BATCH_SIZE, cudaMemcpyHostToDevice));


		// Forward propagation
		ForwardPropagate();

		// Backward propagation
		BackPropagate();

		// Compute learning rate
		float learningRate = static_cast<float>(LEARNING_RATE * pow((1.0 + LR_GAMMA * iter), (-LR_POWER)));
		
		// Update weights
		UpdateWeights(learningRate);
	}
	checkCudaErrors(cudaDeviceSynchronize());
	auto t2 = std::chrono::high_resolution_clock::now();

	printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / iterations);

}


void NeuralNetwork::Test()
{
	float classification_error = 1.0f;

	int classifications = (int)(Data->getTestSize() / 10);

	// Test the resulting neural network's classification

	// Initialize a TrainingContext structure for testing (different batch size)
	NeuralNetwork test_nn;

	// Ensure correct workspaceSize is allocated for testing
	if (WorkspaceSize < test_nn.WorkspaceSize)
	{
		checkCudaErrors(cudaFree(device_workspace));
		checkCudaErrors(cudaMalloc(&device_workspace, test_nn.WorkspaceSize));
	}

	int num_errors = 0;
	for (int i = 0; i < classifications; ++i)
	{
		int output_number = Data->getOutputNumber();
		float *device_data = Data->getData();
		std::vector<float> data(output_number);
		// Normalize image to be in [0,1]
		for (int j = 0; j < output_number; ++j)
			data[j] = (float)Data->TestSet[i * output_number + j] / 255.0f;

		checkCudaErrors(cudaMemcpyAsync(device_data, &data[0], sizeof(float) * output_number, cudaMemcpyHostToDevice));

		// Forward propagate test image
		Predict();

		// Perform classification
		std::vector<float> class_vec(10);

		// Copy back result
		checkCudaErrors(cudaMemcpy(&class_vec[0], Layers.back()->getData(), sizeof(float) * 10, cudaMemcpyDeviceToHost));

		// Determine classification according to maximal response
		int chosen = 0;
		for (int id = 1; id < 10; ++id)
		{
			if (class_vec[chosen] < class_vec[id]) chosen = id;
		}

		if (chosen != Data->TestLabels[i])
			++num_errors;
	}
	classification_error = (float)num_errors / (float)classifications;

	printf("Classification result: %.2f%% error (used %d images)\n", classification_error * 100.0f, (int)classifications);
	
}


void NeuralNetwork::ForwardPropagate()
{
	checkCudaErrors(cudaSetDevice(GPUid));
	
	int count = Layers.size();
	for (int i = 0; i < count; i++)
	{
		(Layers[i])->ForwardPropagate();
	}
}

void NeuralNetwork::Predict()
{
	checkCudaErrors(cudaSetDevice(GPUid));

	int count = Layers.size();
	for (int i = 0; i < count; i++)
	{
		(Layers[i])->Predict();
	}
}

void NeuralNetwork::BackPropagate()
{
	static float alpha = 1.0f, beta = 0.0f;

	for (signed int i = Layers.size() - 1; i >= 0; i--)
	{
		Layers[i]->BackPropagate();
	}
}

void NeuralNetwork::UpdateWeights(float learning_rate)
{
	//float alpha = -learning_rate;

	checkCudaErrors(cudaSetDevice(GPUid));

	int count = Layers.size();
	for (int i = 0; i < count; i++)
	{
		Layers[i]->UpdateWeights(learning_rate);
	}

}

ResidualBlock::ResidualBlock(NeuralNetwork *neuralnetwork, Layer *lastlayer)
{
	neuralNetwork = neuralnetwork;

	
	BranchLayer *branchlayer = new BranchLayer(neuralNetwork, lastlayer);
	device_branch_data = branchlayer->device_data;
	branchlayer->ResBlock = this;
	neuralNetwork->AddLayer(branchlayer);

	neuralNetwork->AddLayer(new ConvolutionLayer(neuralNetwork, branchlayer, 50, 3, 1));
	neuralNetwork->AddLayer(new ActivationLayer(neuralNetwork, neuralNetwork->Layers.back()));
	neuralNetwork->AddLayer(new ConvolutionLayer(neuralNetwork, neuralNetwork->Layers.back(), 50, 3, 1));

	LastLayer = neuralNetwork->Layers.back();
	LastLayer->NextLayer = this;


	OutputNumber = InputNumber = LastLayer->OutputNumber;
	OutputChannels = InputChannels = LastLayer->InputChannels;
	OutputWidth = InputWidth = LastLayer->OutputWidth;
	OutputHeight = InputHeight = LastLayer->OutputHeight;
	KernelSize = 0;
	Padding = 0;
	Stride = 1;

	deviceMalloc(BATCH_SIZE);
	CreateDescriptor(BATCH_SIZE);
}

ResidualBlock::~ResidualBlock()
{
	deviceFree();
	DestroyDescriptor();
}

inline void ResidualBlock::ForwardPropagate() 
{
	static float alpha = 1.0;

	checkCudaErrors(cublasScopy(neuralNetwork->cublasHandle, InputNumber * BATCH_SIZE, 
		LastLayer->device_data, 1, device_data, 1));
	checkCudaErrors(cublasSaxpy(neuralNetwork->cublasHandle, InputNumber * BATCH_SIZE,
		&alpha, device_branch_data, 1, device_data, 1));

}
inline void ResidualBlock::BackPropagate() 
{
	device_diff_data = NextLayer->device_diff_data;

}
inline void ResidualBlock::UpdateWeights(float learning_rate) 
{
	
}

inline void ResidualBlock::deviceMalloc(int batchsize)
{
	checkCudaErrors(cudaMalloc(&device_data, sizeof(float) * batchsize * OutputNumber));
}

inline void ResidualBlock::deviceFree()
{
	checkCudaErrors(cudaFree(device_data));
}

inline void ResidualBlock::CreateDescriptor(int batchsize)
{
	TensorDesc = LastLayer->TensorDesc;
}

inline void ResidualBlock::DestroyDescriptor()
{

}





BranchLayer::BranchLayer(NeuralNetwork *neuralnetwork, Layer *lastlayer)
{
	neuralNetwork = neuralnetwork;
	LastLayer = lastlayer;
	LastLayer->NextLayer = this;

	device_data = LastLayer->device_data;

	OutputNumber = InputNumber = lastlayer->OutputNumber;
	OutputChannels = InputChannels = lastlayer->InputChannels;
	OutputWidth = InputWidth = lastlayer->OutputWidth;
	OutputHeight = InputHeight = lastlayer->OutputHeight;
	KernelSize = 0;
	Padding = 0;
	Stride = 1;

	deviceMalloc(BATCH_SIZE);
	CreateDescriptor(BATCH_SIZE);
}

BranchLayer::~BranchLayer()
{
	deviceFree();
	DestroyDescriptor();
}

inline void BranchLayer::ForwardPropagate()
{
	device_data = LastLayer->device_data;
}
inline void BranchLayer::BackPropagate()
{
	static float alpha = 1.0;
	checkCudaErrors(cublasScopy(neuralNetwork->cublasHandle, InputNumber,
		NextLayer->device_diff_data, 1, device_diff_data, 1));
	if (ResBlock != nullptr)
	{
		checkCudaErrors(cublasSaxpy(neuralNetwork->cublasHandle, InputNumber,
			&alpha, ResBlock->device_diff_data, 1, device_diff_data, 1));
	}
}
inline void BranchLayer::UpdateWeights(float learning_rate)
{

}

inline void BranchLayer::deviceMalloc(int batchsize)
{
	//device_data = LastLayer->device_data;
	checkCudaErrors(cudaMalloc(&device_diff_data, sizeof(float) * batchsize * InputNumber));
}

inline void BranchLayer::deviceFree()
{
	checkCudaErrors(cudaFree(device_diff_data));
}

inline void BranchLayer::CreateDescriptor(int batchsize)
{
	TensorDesc = LastLayer->TensorDesc;
}

inline void BranchLayer::DestroyDescriptor()
{

}



bool Layer::FromFile(const char *fileprefix)
{
	std::stringstream ssf, ssbf;
	ssf << fileprefix << ".bin";
	ssbf << fileprefix << ".bias.bin";

	// Read weights file
	FILE *fp = fopen(ssf.str().c_str(), "rb");
	if (!fp)
	{
		printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
		return false;
	}
	fread(&ParamW[0], sizeof(float), ParamW.size(), fp);
	fclose(fp);

	// Read bias file
	fp = fopen(ssbf.str().c_str(), "rb");
	if (!fp)
	{
		printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
		return false;
	}
	fread(&ParamB[0], sizeof(float), ParamB.size(), fp);
	fclose(fp);
	return true;
}

void Layer::ToFile(const char *fileprefix)
{
	std::stringstream ssf, ssbf;
	ssf << fileprefix << ".bin";
	ssbf << fileprefix << ".bias.bin";

	// Write weights file
	FILE *fp = fopen(ssf.str().c_str(), "wb");
	if (!fp)
	{
		printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
		exit(2);
	}
	fwrite(&ParamW[0], sizeof(float), ParamW.size(), fp);
	fclose(fp);

	// Write bias file
	fp = fopen(ssbf.str().c_str(), "wb");
	if (!fp)
	{
		printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
		exit(2);
	}
	fwrite(&ParamB[0], sizeof(float), ParamB.size(), fp);
	fclose(fp);
}





BatchNormLayer::BatchNormLayer(NeuralNetwork *neuralnetwork, Layer *lastlayer)
{
	neuralNetwork = neuralnetwork;
	LastLayer = lastlayer;
	LastLayer->NextLayer = this;

	OutputNumber = InputNumber = lastlayer->OutputNumber;
	OutputChannels = InputChannels = lastlayer->InputChannels;
	OutputWidth = InputWidth = lastlayer->OutputWidth;
	OutputHeight = InputHeight = lastlayer->OutputHeight;
	KernelSize = 0;
	Padding = 0;
	Stride = 1;	

	deviceMalloc(BATCH_SIZE);
	CreateDescriptor(BATCH_SIZE);
}

BatchNormLayer::~BatchNormLayer()
{
	deviceFree();
	DestroyDescriptor();
}

inline void BatchNormLayer::ForwardPropagate()
{
	static float alpha = 1.0, beta = 0.0;
	device_data = LastLayer->device_data;

	//checkCudaErrors(cudnnBatchNormalizationForwardTraining(neuralNetwork->cudnnHandle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, 
	//	&alpha, &beta, LastLayer->TensorDesc, LastLayer->device_data, TensorDesc, device_data,
	//	));
}

inline void BatchNormLayer::BackPropagate()
{
	device_diff_data = NextLayer->device_diff_data;
}

inline void BatchNormLayer::Predict()
{

}

inline void BatchNormLayer::UpdateWeights(float learning_rate)
{

}

inline void BatchNormLayer::deviceMalloc(int batchsize)
{
	device_data = LastLayer->device_data;
}

inline void BatchNormLayer::deviceFree()
{

}

inline void BatchNormLayer::CreateDescriptor(int batchsize)
{
	checkCUDNN(cudnnSetTensor4dDescriptor(TensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchsize, OutputChannels, OutputHeight, OutputWidth));
	checkCUDNN(cudnnSetTensor4dDescriptor(ParamBTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchsize, OutputChannels, OutputHeight, OutputWidth));
}

inline void BatchNormLayer::DestroyDescriptor()
{
	checkCUDNN(cudnnDestroyTensorDescriptor(TensorDesc));
}