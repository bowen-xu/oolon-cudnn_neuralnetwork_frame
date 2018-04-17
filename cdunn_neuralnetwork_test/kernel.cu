#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>

#include <stdio.h>
#include "neuralnetwork.cuh"

void main()
{
	NeuralNetwork nn;
	DataSet *data = new DataSet();
	data->LoadData("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	
	nn.AddData(data);

	nn.AddLayer(new ConvolutionLayer(&nn, data, "Conv1", 20, 5), true);
	nn.AddLayer(new MaxPoolLayer(&nn, nn.Layers.back(), "Pool1", 2, 2));
	nn.AddLayer(new ConvolutionLayer(&nn, nn.Layers.back(), "Conv2", 50, 3));
	nn.AddLayer(new MaxPoolLayer(&nn, nn.Layers.back(), "Pool2", 2, 2));

	//nn.AddLayer(new ConvolutionLayer(&nn, nn.Layers.back(), "Conv3", 50, 3, 1));
	//nn.AddLayer(new ActivationLayer(&nn, nn.Layers.back(), "Pool3"));
	//nn.AddLayer(new ConvolutionLayer(&nn, nn.Layers.back(), "Conv4", 50, 3, 1));
	//nn.AddLayer(new ActivationLayer(&nn, nn.Layers.back(), "Pool4"));
	//nn.AddLayer(new ConvolutionLayer(&nn, nn.Layers.back(), "Conv5", 50, 3, 1));
	//nn.AddLayer(new ActivationLayer(&nn, nn.Layers.back(), "Pool5"));
	//nn.AddLayer(new ConvolutionLayer(&nn, nn.Layers.back(), "Conv6", 50, 3, 1));
	//nn.AddLayer(new ActivationLayer(&nn, nn.Layers.back(), "Pool6"));
	//nn.AddLayer(new ConvolutionLayer(&nn, nn.Layers.back(), "Conv7", 50, 3, 1));
	//nn.AddLayer(new ActivationLayer(&nn, nn.Layers.back(), "Pool7"));
	//nn.AddLayer(new ConvolutionLayer(&nn, nn.Layers.back(), "Conv8", 50, 3, 1));
	//nn.AddLayer(new ActivationLayer(&nn, nn.Layers.back(), "Pool8"));
	//nn.AddLayer(new ConvolutionLayer(&nn, nn.Layers.back(), "Conv9", 50, 3, 1));
	//nn.AddLayer(new ActivationLayer(&nn, nn.Layers.back(), "Pool9"));
	//nn.AddLayer(new ConvolutionLayer(&nn, nn.Layers.back(), "Conv10", 50, 3, 1));
	//nn.AddLayer(new ActivationLayer(&nn, nn.Layers.back(), "Pool10"));

	Layer *layerbuf = nn.Layers.back();
	nn.AddLayer(new FullyConnectedLayer(&nn, nn.Layers.back(), "FC1", 500));
	nn.AddLayer(new ActivationLayer(&nn, nn.Layers.back(), "ReLU1"));
	nn.AddLayer(new FullyConnectedLayer(&nn, nn.Layers.back(), "FC2", 10));
	nn.AddLayer(new OutputLayer(&nn, nn.Layers.back(), "Softmax1", data->getLabels()));

	nn.AddLayer(new FullyConnectedLayer(&nn, layerbuf, "FC3", 500));
	nn.AddLayer(new ActivationLayer(&nn, nn.Layers.back(), "ReLU2"));
	nn.AddLayer(new FullyConnectedLayer(&nn, nn.Layers.back(), "FC4", 100));
	nn.AddLayer(new ActivationLayer(&nn, nn.Layers.back(), "ReLU3"));
	nn.AddLayer(new FullyConnectedLayer(&nn, nn.Layers.back(), "FC5", 10));
	nn.AddLayer(new OutputLayer(&nn, nn.Layers.back(), "Softmax2", data->getLabels()));
	
	nn.Create();
	//nn.Load("iteration2000");
	nn.Train(1000);
	nn.Save("iteration2000");
	nn.Test();
	nn.Destroy();

	getchar();

}