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
__global__ void SoftmaxLossBackprop(const float *label, int num_labels, int batch_size, float *diff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batch_size)
		return;

	const int label_value = static_cast<int>(label[idx]);

	// For each item in the batch, decrease the result of the label's value by 1
	diff[idx * num_labels + label_value] -= 1.0f;
}

/*
// FullyConnectedLayer
*/
FullyConnectedLayer::FullyConnectedLayer(int input_num, int output_num)
{
	InputNumber = input_num;
	OutputNumber = output_num;
	ParamW.resize(input_num * output_num);
	ParamB.resize(output_num);

	random_device rd;
	mt19937 gen(rd());
	float wfc = sqrt(3.0f / (InputNumber * OutputNumber));
	std::uniform_real_distribution<> dfc(-wfc, wfc);
	for (auto&& iter : ParamW)
		iter = static_cast<float>(dfc(gen));
	for (auto&& iter : ParamB)
		iter = static_cast<float>(dfc(gen));

	CreateDescriptor(BATCH_SIZE);
	deviceMalloc(BATCH_SIZE);
}

FullyConnectedLayer::~FullyConnectedLayer()
{
	DestroyDescriptor();
	deviceFree();
}


inline void FullyConnectedLayer::deviceMalloc(int batchsize)
{
	// 前向传播数据
	checkCudaErrors(cudaMalloc(&device_data, sizeof(float) * batchsize * OutputNumber));	// GPU中给数据开辟空间
																							// 参数
	checkCudaErrors(cudaMalloc(&device_param_w, sizeof(float) * ParamW.size()));			// GPU中给参数w开辟空间
	checkCudaErrors(cudaMalloc(&device_param_b, sizeof(float) * ParamB.size()));			// GPU中给参数b开辟空间
																							// 梯度
	checkCudaErrors(cudaMalloc(&device_grad_w, sizeof(float) * ParamW.size()));			// GPU中给梯度w开辟空间
	checkCudaErrors(cudaMalloc(&device_grad_b, sizeof(float) * ParamB.size()));			// GPU中给梯度b开辟空间
																						// 反向传播数据
	checkCudaErrors(cudaMalloc(&device_diff_data, sizeof(float) * batchsize * InputNumber));

	// 拷贝初始化数据到GPU
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
ActivationLayer::ActivationLayer(int num, cudnnActivationMode_t mode, cudnnNanPropagation_t nanopt, double coef)
{
	Number = num;
	ActivationMode = mode;
	NanOption = nanopt;
	Coef = coef;

	CreateDescriptor(BATCH_SIZE);
	deviceMalloc(BATCH_SIZE);
}

ActivationLayer::~ActivationLayer()
{
	DestroyDescriptor();
	deviceFree();
}

inline void ActivationLayer::deviceMalloc(int batchsize)
{
	// 前向传播数据
	checkCudaErrors(cudaMalloc(&device_data, sizeof(float) * batchsize * Number));
	// 反向传播数据
	checkCudaErrors(cudaMalloc(&device_diff_data, sizeof(float) * batchsize * Number));
}

inline void ActivationLayer::deviceFree()
{
	checkCudaErrors(cudaFree(device_data));
	checkCudaErrors(cudaFree(device_diff_data));
}

inline void ActivationLayer::CreateDescriptor(int batchsize)
{
	// 创建描述器
	checkCUDNN(cudnnCreateActivationDescriptor(&ActivationDesc));

	// 设置描述器
	checkCUDNN(cudnnSetActivationDescriptor(ActivationDesc, ActivationMode, NanOption, Coef));
}

inline void ActivationLayer::DestroyDescriptor()
{
	checkCUDNN(cudnnDestroyActivationDescriptor(ActivationDesc));
}



/*
// ConvolutionLayer
*/
ConvolutionLayer::ConvolutionLayer(cudnnHandle_t cudnnhandle, cudnnTensorDescriptor_t lastTensorDesc, int in_channels, int out_channels, int kernel_size, int in_width, int in_height, int padding, int stride)
{
	InputChannels = in_channels;
	OutputChannels = out_channels;
	KernelSize = kernel_size;
	InputWidth = in_width;
	InputHeight = in_height;
	Padding = padding;
	Stride = stride;

	OutputHeight = (InputHeight + 2 * Padding - KernelSize) / Stride + 1;
	OutputWidth = (InputWidth + 2 * Padding - KernelSize) / Stride + 1;
	ParamW.resize(in_channels * kernel_size * kernel_size * out_channels);
	ParamB.resize(out_channels);

	LastTensorDesc = lastTensorDesc;
	cudnnHandle = cudnnhandle;

	random_device rd;
	mt19937 gen(rd());
	float wconv = sqrt(3.0f / (KernelSize * KernelSize * InputChannels));
	std::uniform_real_distribution<> dconv(-wconv, wconv);
	for (auto&& iter : ParamW)
		iter = static_cast<float>(dconv(gen));
	for (auto&& iter : ParamB)
		iter = static_cast<float>(dconv(gen));

	CreateDescriptor(BATCH_SIZE);
	deviceMalloc(BATCH_SIZE);

}

ConvolutionLayer::~ConvolutionLayer()
{
	DestroyDescriptor();
	deviceFree();
}


inline void ConvolutionLayer::deviceMalloc(int batchsize)
{
	// 前向传播数据
	checkCudaErrors(cudaMalloc(&device_data, sizeof(float) * batchsize * OutputChannels * OutputHeight * OutputWidth));

	// 参数
	checkCudaErrors(cudaMalloc(&device_param_w, sizeof(float) * ParamW.size()));
	checkCudaErrors(cudaMalloc(&device_param_b, sizeof(float) * ParamB.size()));
	// 梯度
	checkCudaErrors(cudaMalloc(&device_grad_w, sizeof(float) * ParamW.size()));
	checkCudaErrors(cudaMalloc(&device_grad_b, sizeof(float) * ParamB.size()));
	// 反向传播数据
	checkCudaErrors(cudaMalloc(&device_diff_data, sizeof(float) * batchsize * OutputChannels * InputWidth * InputHeight));

	// 拷贝初始化数据到GPU
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
	// 创建张量
	checkCUDNN(cudnnCreateTensorDescriptor(&TensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&BiasTensorDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&FilterDesc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&ConvDesc));

	// 设置张量
	checkCUDNN(cudnnSetTensor4dDescriptor(BiasTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, OutputChannels, 1, 1));
	checkCUDNN(cudnnSetFilter4dDescriptor(FilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OutputChannels, InputChannels, KernelSize, KernelSize));
	checkCUDNN(cudnnSetConvolution2dDescriptor(ConvDesc, Padding, Padding, Stride, Stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
	checkCUDNN(cudnnSetTensor4dDescriptor(TensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchsize, OutputChannels, OutputHeight, OutputWidth));

	// 前向传播算法设置
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, LastTensorDesc, FilterDesc, ConvDesc, TensorDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &FwdAlgDesc));
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, LastTensorDesc, FilterDesc, ConvDesc, TensorDesc, FwdAlgDesc, &tempsize));
	WorkspaceSize = max(WorkspaceSize, tempsize);

	// 反向传播算法设置
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle, LastTensorDesc, TensorDesc, ConvDesc, FilterDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &BwdAlgDesc));
	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, LastTensorDesc, TensorDesc, ConvDesc, FilterDesc, BwdAlgDesc, &tempsize));
	WorkspaceSize = max(WorkspaceSize, tempsize);
	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle, FilterDesc, TensorDesc, ConvDesc, LastTensorDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &BwdDataAlgDesc));
	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, FilterDesc, TensorDesc, ConvDesc, LastTensorDesc, BwdDataAlgDesc, &tempsize));
	WorkspaceSize = max(WorkspaceSize, tempsize);
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
MaxPoolLayer::MaxPoolLayer(int size, int stride, ConvolutionLayer &lastConv)
{
	Size = size;
	Stride = stride;
	OutputChannels = lastConv.OutputChannels;
	OutputWidth = (lastConv.OutputWidth / stride);
	OutputHeight = (lastConv.OutputHeight / stride);

	CreateDescriptor(BATCH_SIZE);
	deviceMalloc(BATCH_SIZE);
}

MaxPoolLayer::~MaxPoolLayer()
{
	DestroyDescriptor();
	deviceFree();
}


inline void MaxPoolLayer::deviceMalloc(int batchsize)
{
	// 前向传播数据
	checkCudaErrors(cudaMalloc(&device_data, sizeof(float) * batchsize * OutputChannels * OutputHeight * OutputWidth));
	// 反向传播数据
	checkCudaErrors(cudaMalloc(&device_diff_data, sizeof(float) * batchsize * OutputChannels * OutputHeight * OutputWidth));
}

inline void MaxPoolLayer::deviceFree()
{
	checkCudaErrors(cudaFree(device_data));
	checkCudaErrors(cudaFree(device_diff_data));
}

inline void MaxPoolLayer::CreateDescriptor(int batchsize)
{
	// 创建描述器
	checkCUDNN(cudnnCreateTensorDescriptor(&TensorDesc));
	checkCUDNN(cudnnCreatePoolingDescriptor(&PoolDesc));

	// 设置描述器
	checkCUDNN(cudnnSetPooling2dDescriptor(PoolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, Size, Size, 0, 0, Stride, Stride));
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
	printf("Reading input data\n");

	size_t train_size = ReadUByteDataset(TrainingSetName.c_str(), TrainingLabelsName.c_str(), nullptr, nullptr, Width, Height);
	size_t test_size = ReadUByteDataset(TestSetName.c_str(), TestLabelsName.c_str(), nullptr, nullptr, Width, Height);
	TrainSize = train_size;
	TestSize = test_size;
	if (train_size == 0)
		exit(1);

	TrainSet.resize(train_size * Width * Height * Channels);
	TrainLabels.resize(train_size);
	TestSet.resize(test_size * Width * Height * Channels);
	TestLabels.resize(test_size);

	if (ReadUByteDataset(TrainingSetName.c_str(), TrainingLabelsName.c_str(), &TrainSet[0], &TrainLabels[0], Width, Height) != train_size)
		exit(2);
	if (ReadUByteDataset(TestSetName.c_str(), TestLabelsName.c_str(), &TestSet[0], &TestLabels[0], Width, Height) != test_size)
		exit(3);

	printf("Done. Training dataset size: %d, Test dataset size: %d\n", (int)train_size, (int)test_size);
	
	// Normalize training set to be in [0,1]
	printf("Normalizing training data...\n");
	TrainSet_float.resize(TrainSet.size());
	TrainLabels_float.resize(TrainLabels.size());
	for (size_t i = 0; i < train_size * Channels * Width * Height; ++i)
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

inline void DataSet::deviceMalloc(int batchsize)
{
	// 前向传播数据
	checkCudaErrors(cudaMalloc(&device_data, sizeof(float) * batchsize * Channels * Height * Width));
	checkCudaErrors(cudaMalloc(&device_labels, sizeof(float) * batchsize));
}

inline void DataSet::deviceFree()
{
	checkCudaErrors(cudaFree(device_data));
	checkCudaErrors(cudaFree(device_labels));
}

inline void DataSet::CreateDescriptor(int batchsize)
{
	// 创建张量
	checkCUDNN(cudnnCreateTensorDescriptor(&TensorDesc)); //

														  // 设置张量
	checkCUDNN(cudnnSetTensor4dDescriptor(TensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchsize, Channels, Height, Width));
}

inline void DataSet::DestroyDescriptor()
{
	checkCUDNN(cudnnDestroyTensorDescriptor(TensorDesc));
}

/*
// OutputLayer
*/
OutputLayer::OutputLayer(int num)
{
	Number = num;

	CreateDescriptor(BATCH_SIZE);
	deviceMalloc(BATCH_SIZE);
}

OutputLayer::~OutputLayer()
{
	DestroyDescriptor();
	deviceFree();
}

inline void OutputLayer::deviceMalloc(int batchsize)
{
	// 前向传播数据
	checkCudaErrors(cudaMalloc(&device_data, sizeof(float) * batchsize * Number));
	// 反向传播数据
	checkCudaErrors(cudaMalloc(&device_diff_data, sizeof(float) * batchsize * Number));
	checkCudaErrors(cudaMalloc(&device_loss_data, sizeof(float) * batchsize * Number));
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

void NeuralNetwork::Create()
{
	Image = new DataSet();
	Conv1 = new ConvolutionLayer(cudnnHandle, Image->TensorDesc, Image->Channels, 20, 5, Image->Height, Image->Width);
	Pool1 = new MaxPoolLayer(2, 2, *Conv1);
	Conv2 = new ConvolutionLayer(cudnnHandle, Pool1->TensorDesc, Conv1->OutputChannels, 50, 5, Conv1->OutputWidth / Pool1->Stride, Conv1->OutputHeight / Pool1->Stride);
	Pool2 = new MaxPoolLayer(2, 2, *Conv2);
	FC1 = new FullyConnectedLayer((Conv2->OutputChannels * Conv2->OutputWidth * Conv2->OutputHeight) / (Pool2->Stride * Pool2->Stride), 500);
	ACTN1 = new ActivationLayer(FC1->OutputNumber);
	FC2 = new FullyConnectedLayer(FC1->OutputNumber, 10);
	RSLT = new OutputLayer(FC2->OutputNumber);
	
	checkCudaErrors(cudaMalloc(&device_ones, sizeof(float)* BATCH_SIZE));
	FillOnes <<<RoundUp(BATCH_SIZE, BW), BW>>> (device_ones, BATCH_SIZE);
	WorkspaceSize = max(Conv1->WorkspaceSize, Conv2->WorkspaceSize);
	if (WorkspaceSize > 0)
		checkCudaErrors(cudaMalloc(&device_workspace, WorkspaceSize));

}

void NeuralNetwork::Destroy()
{
	delete Image;
	delete Conv1;
	delete Pool1;
	delete Conv2;
	delete Pool2;
	delete FC1;
	delete ACTN1;
	delete FC2;
	delete RSLT;

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
	size_t train_size = Image->TrainSize;
	for (int iter = 0; iter < iterations; ++iter)
	{
		// Train
		int imageid = iter % (train_size / BATCH_SIZE);

		// Prepare current batch on device
		checkCudaErrors(cudaMemcpyAsync(Image->device_data, &((Image->TrainSet_float)[imageid * BATCH_SIZE * Image->Width * Image->Height * Image->Channels]),
			sizeof(float) * BATCH_SIZE * Image->Width *Image->Height * Image->Channels, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyAsync(Image->device_labels, &((Image->TrainLabels_float)[imageid * BATCH_SIZE]),
			sizeof(float) * BATCH_SIZE, cudaMemcpyHostToDevice));

		// Forward propagation
		ForwardPropagate();

		//std::vector<float> class_vec(10);
		//// Copy back result
		//checkCudaErrors(cudaMemcpy(&class_vec[0], RSLT->device_data, sizeof(float) * 10, cudaMemcpyDeviceToHost));


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

	int classifications = (int)(Image->TestSize);

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
		std::vector<float> data(Image->Width * Image->Height);
		// Normalize image to be in [0,1]
		for (int j = 0; j < Image->Width * Image->Height; ++j)
			data[j] = (float)Image->TestSet[i * Image->Width * Image->Height * Image->Channels + j] / 255.0f;

		checkCudaErrors(cudaMemcpyAsync(Image->device_data, &data[0], sizeof(float) * Image->Width * Image->Height, cudaMemcpyHostToDevice));

		// Forward propagate test image
		ForwardPropagate();

		// Perform classification
		std::vector<float> class_vec(10);

		// Copy back result
		checkCudaErrors(cudaMemcpy(&class_vec[0], RSLT->device_data, sizeof(float) * 10, cudaMemcpyDeviceToHost));

		// Determine classification according to maximal response
		int chosen = 0;
		for (int id = 1; id < 10; ++id)
		{
			if (class_vec[chosen] < class_vec[id]) chosen = id;
		}

		if (chosen != Image->TestLabels[i])
			++num_errors;
	}
	classification_error = (float)num_errors / (float)classifications;

	printf("Classification result: %.2f%% error (used %d images)\n", classification_error * 100.0f, (int)classifications);
	
}


void NeuralNetwork::ForwardPropagate()
{
	float alpha = 1.0f, beta = 0.0f;
	checkCudaErrors(cudaSetDevice(GPUid));

	// Conv1 layer
	checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, Image->TensorDesc,
		Image->device_data, Conv1->FilterDesc, Conv1->device_param_w, Conv1->ConvDesc,
		Conv1->FwdAlgDesc, device_workspace, WorkspaceSize, &beta,
		Conv1->TensorDesc, Conv1->device_data));

	checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, Conv1->BiasTensorDesc,
		Conv1->device_param_b, &alpha, Conv1->TensorDesc, Conv1->device_data));

	// Pool1 layer
	checkCUDNN(cudnnPoolingForward(cudnnHandle, Pool1->PoolDesc, &alpha, Conv1->TensorDesc,
		Conv1->device_data, &beta, Pool1->TensorDesc, Pool1->device_data));

	// Conv2 layer
	checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, Pool1->TensorDesc,
		Pool1->device_data, Conv2->FilterDesc, Conv2->device_param_w, Conv2->ConvDesc,
		Conv2->FwdAlgDesc, device_workspace, WorkspaceSize, &beta,
		Conv2->TensorDesc, Conv2->device_data));
	checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, Conv2->BiasTensorDesc,
		Conv2->device_param_b, &alpha, Conv2->TensorDesc, Conv2->device_data));

	// Pool2 layer
	checkCUDNN(cudnnPoolingForward(cudnnHandle, Pool2->PoolDesc, &alpha, Conv2->TensorDesc,
		Conv2->device_data, &beta, Pool2->TensorDesc, Pool2->device_data));

	// FC1 layer
	// Forward propagate neurons using weights (fc1 = pfc1'*pool2)
	checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
		FC1->OutputNumber, BATCH_SIZE, FC1->InputNumber,
		&alpha,
		FC1->device_param_w, FC1->InputNumber,
		Pool2->device_data, FC1->InputNumber,
		&beta,
		FC1->device_data, FC1->OutputNumber));
	// Add bias using GEMM's "beta" (fc1 += pfc1bias*1_vec')
	checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		FC1->OutputNumber, BATCH_SIZE, 1,
		&alpha,
		FC1->device_param_b, FC1->OutputNumber,
		device_ones, 1,
		&alpha,
		FC1->device_data, FC1->OutputNumber));

	// ReLU activation
	checkCUDNN(cudnnActivationForward(cudnnHandle, ACTN1->ActivationDesc, &alpha,
		FC1->TensorDesc, FC1->device_data, &beta, FC1->TensorDesc, ACTN1->device_data));

	// FC2 layer
	// Forward propagate neurons using weights (fc2 = pfc2'*fc1relu)
	checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
		FC2->OutputNumber, BATCH_SIZE, FC2->InputNumber,
		&alpha,
		FC2->device_param_w, FC2->InputNumber,
		ACTN1->device_data, FC2->InputNumber,
		&beta,
		FC2->device_data, FC2->OutputNumber));
	// Add bias using GEMM's "beta" (fc2 += pfc2bias*1_vec')
	checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		FC2->OutputNumber, BATCH_SIZE, 1,
		&alpha,
		FC2->device_param_b, FC2->OutputNumber,
		device_ones, 1,
		&alpha,
		FC2->device_data, FC2->OutputNumber));

	// Softmax loss
	checkCUDNN(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
		&alpha, FC2->TensorDesc, FC2->device_data, &beta, FC2->TensorDesc, RSLT->device_data));
}

void NeuralNetwork::BackPropagate()
{
	float alpha = 1.0f, beta = 0.0f;

	float scalVal = 1.0f / static_cast<float>(BATCH_SIZE);

	checkCudaErrors(cudaSetDevice(GPUid));

	// Initialization (using the training error function)
	checkCudaErrors(cudaMemcpyAsync(RSLT->device_diff_data, RSLT->device_data, sizeof(float) * BATCH_SIZE * FC2->OutputNumber, cudaMemcpyDeviceToDevice));

	// Softmax layer
	SoftmaxLossBackprop <<<RoundUp(BATCH_SIZE, BW), BW >>> (Image->device_labels, FC2->OutputNumber, BATCH_SIZE, RSLT->device_diff_data);

	// Accounting for batch size in SGD
	checkCudaErrors(cublasSscal(cublasHandle, FC2->OutputNumber * BATCH_SIZE, &scalVal, RSLT->device_diff_data, 1));

	// FC2 layer
	// Compute derivative with respect to weights: gfc2 = (fc1relu * dfc2smax')
	checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, FC2->InputNumber, FC2->OutputNumber, BATCH_SIZE,
		&alpha, ACTN1->device_data, FC2->InputNumber, RSLT->device_diff_data, FC2->OutputNumber, &beta, FC2->device_grad_w, FC2->InputNumber));
	// Compute derivative with respect to bias: gfc2bias = dfc2smax * 1_vec
	checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, FC2->OutputNumber, BATCH_SIZE,
		&alpha, RSLT->device_diff_data, FC2->OutputNumber, device_ones, 1, &beta, FC2->device_grad_b, 1));
	// Compute derivative with respect to data (for previous layer): pfc2*dfc2smax (500x10*10xN)
	checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, FC2->InputNumber, BATCH_SIZE, FC2->OutputNumber,
		&alpha, FC2->device_param_w, FC2->InputNumber, RSLT->device_diff_data, FC2->OutputNumber, &beta, FC2->device_diff_data, FC2->InputNumber));

	// ReLU activation
	checkCUDNN(cudnnActivationBackward(cudnnHandle, ACTN1->ActivationDesc, &alpha,
		FC1->TensorDesc, ACTN1->device_data, FC1->TensorDesc, FC2->device_diff_data,
		FC1->TensorDesc, FC1->device_data, &beta, FC1->TensorDesc, ACTN1->device_diff_data));

	// FC1 layer
	// Compute derivative with respect to weights: gfc1 = (pool2 * dfc1relu')
	checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, FC1->InputNumber, FC1->OutputNumber, BATCH_SIZE,
		&alpha, Pool2->device_data, FC1->InputNumber, ACTN1->device_diff_data, FC1->OutputNumber, &beta, FC1->device_grad_w, FC1->InputNumber));
	// Compute derivative with respect to bias: gfc1bias = dfc1relu * 1_vec
	checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, FC1->OutputNumber, BATCH_SIZE,
		&alpha, ACTN1->device_diff_data, FC1->OutputNumber, device_ones, 1, &beta, FC1->device_grad_b, 1));
	// Compute derivative with respect to data (for previous layer): pfc1*dfc1relu (800x500*500xN)
	checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, FC1->InputNumber, BATCH_SIZE, FC1->OutputNumber,
		&alpha, FC1->device_param_w, FC1->InputNumber, ACTN1->device_diff_data, FC1->OutputNumber, &beta, FC1->device_diff_data, FC1->InputNumber));

	// Pool2 layer
	checkCUDNN(cudnnPoolingBackward(cudnnHandle, Pool2->PoolDesc, &alpha,
		Pool2->TensorDesc, Pool2->device_data, Pool2->TensorDesc, FC1->device_diff_data,
		Conv2->TensorDesc, Conv2->device_data, &beta, Conv2->TensorDesc, Pool2->device_diff_data));

	// Conv2 layer
	checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, Conv2->TensorDesc,
		Pool2->device_diff_data, &beta, Conv2->BiasTensorDesc, Conv2->device_grad_b));


	checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, Pool1->TensorDesc,
		Pool1->device_data, Conv2->TensorDesc, Pool2->device_diff_data, Conv2->ConvDesc,
		Conv2->BwdAlgDesc, device_workspace, WorkspaceSize,
		&beta, Conv2->FilterDesc, Conv2->device_grad_w));

	checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle, &alpha, Conv2->FilterDesc,
		Conv2->device_param_w, Conv2->TensorDesc, Pool2->device_diff_data, Conv2->ConvDesc,
		Conv2->BwdDataAlgDesc, device_workspace, WorkspaceSize,
		&beta, Pool1->TensorDesc, Conv2->device_diff_data));

	// Pool1 layer
	checkCUDNN(cudnnPoolingBackward(cudnnHandle, Pool1->PoolDesc, &alpha,
		Pool1->TensorDesc, Pool1->device_data, Pool1->TensorDesc, Conv2->device_diff_data,
		Conv1->TensorDesc, Conv1->device_data, &beta, Conv1->TensorDesc, Pool1->device_diff_data));

	// Conv1 layer
	checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, Conv1->TensorDesc,
		Pool1->device_diff_data, &beta, Conv1->BiasTensorDesc, Conv1->device_grad_b));

	checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, Image->TensorDesc,
		Image->device_data, Conv1->TensorDesc, Pool1->device_diff_data, Conv1->ConvDesc,
		Conv1->BwdAlgDesc, device_workspace, WorkspaceSize,
		&beta, Conv1->FilterDesc, Conv1->device_grad_w));

	// No need for convBackwardData because there are no more layers below
}

void NeuralNetwork::UpdateWeights(float learning_rate)
{
	float alpha = -learning_rate;

	checkCudaErrors(cudaSetDevice(GPUid));

	// Conv1
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(Conv1->ParamW.size()),
		&alpha, Conv1->device_grad_w, 1, Conv1->device_param_w, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(Conv1->ParamB.size()),
		&alpha, Conv1->device_grad_b, 1, Conv1->device_param_b, 1));

	// Conv2
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(Conv2->ParamW.size()),
		&alpha, Conv2->device_grad_w, 1, Conv2->device_param_w, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(Conv2->ParamB.size()),
		&alpha, Conv2->device_grad_b, 1, Conv2->device_param_b, 1));

	// Fully connected 1
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(FC1->ParamW.size()),
		&alpha, FC1->device_grad_w, 1, FC1->device_param_w, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(FC1->ParamB.size()),
		&alpha, FC1->device_grad_b, 1, FC1->device_param_b, 1));

	// Fully connected 2
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(FC2->ParamW.size()),
		&alpha, FC2->device_grad_w, 1, FC2->device_param_w, 1));
	checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(FC2->ParamB.size()),
		&alpha, FC2->device_grad_b, 1, FC2->device_param_b, 1));
}





