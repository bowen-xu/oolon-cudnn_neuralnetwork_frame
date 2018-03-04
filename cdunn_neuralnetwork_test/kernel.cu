#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>

#include <stdio.h>
#include "neuralnetwork.cuh"

void main()
{
	NeuralNetwork nn;

	nn.AddData(new DataSet());
	nn.AddLayer(new ConvolutionLayer(&nn, nn.Data, 20, 5));
	nn.AddLayer(new MaxPoolLayer(&nn, nn.Layers.back(), 2, 2));
	nn.AddLayer(new ConvolutionLayer(&nn, nn.Layers.back(), 50, 5));
	nn.AddLayer(new MaxPoolLayer(&nn, nn.Layers.back(), 2, 2));
	nn.AddLayer(new FullyConnectedLayer(&nn, nn.Layers.back(), 500));
	nn.AddLayer(new ActivationLayer(&nn, nn.Layers.back()));
	nn.AddLayer(new FullyConnectedLayer(&nn, nn.Layers.back(), 10));
	nn.AddLayer(new OutputLayer(&nn, nn.Layers.back()));

	nn.Create();
	nn.Train(1000);
	nn.Test();
	
	getchar();

}