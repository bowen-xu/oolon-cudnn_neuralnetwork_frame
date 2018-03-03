#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>

#include <stdio.h>
#include "neuralnetwork.cuh"

void main()
{
	NeuralNetwork nn;

	nn.Create();
	nn.Train(10000);
	nn.Test();
	
	getchar();

}