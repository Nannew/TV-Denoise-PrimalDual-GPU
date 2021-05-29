#pragma once


#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/core/cuda/common.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

class TVDenoiseGPU
{
public:
	TVDenoiseGPU();
	~TVDenoiseGPU();

	//TV-L1 image denoising with the primal-dual algorithm.
	//Version 1: GPU by CUDA kernels
	void TV_Denoise_GPU(const cv::cuda::GpuMat& observations, cv::cuda::GpuMat& X, float clambda, int niters);

	//Version 2: CPU
	void TV_Denoise_CPU(cv::Mat& observations, cv::Mat& X, float clambda, int niters);

};