//Image denoising using the TV-L1 model optimized with a primal-dual algorithm.
//The algorithm is implemented on GPU with CUDA and compare the performance with CPU implementation
//https://www.mathworks.com/matlabcentral/fileexchange/57604-tv-l1-image-denoising-algorithm

#pragma once
#define COMPILER_MSVC
#define NOMINMAX

#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>

//Include algorithm headers
#include "TVDenoiseGPU.h"

int main()
{
	TVDenoiseGPU *oTV = new TVDenoiseGPU();

	std::string imgpath("D:\\Workspace\\TV-Denoise-PrimalDual-GPU-Build\\bin\\Release\\lenna_512_noisy.bmp");

	//Use OpenCV gpuMat to hold the data, so no need to free memory after use
	//Read image with OpenCV
	cv::Mat im_noisy;
	im_noisy = cv::imread(imgpath, 0);

	//convert to float
	im_noisy.convertTo(im_noisy, CV_32FC1);


	//--------------------------------------------------------------------------------------------------------
	//Test GPU version
	cv::cuda::GpuMat im_denoised_gpu;
	cv::Mat im_denoised;
	cv::cuda::GpuMat im_noisy_gpu;

	std::chrono::steady_clock::time_point session_begin = std::chrono::steady_clock::now();
	//Upload image from CPU to GPU
	im_noisy_gpu.upload(im_noisy);
	oTV->TV_Denoise_GPU(im_noisy_gpu, im_denoised_gpu, 0.1f, 1000);
	//Download result to CPU
	im_denoised_gpu.download(im_denoised);
	std::chrono::steady_clock::time_point session_end = std::chrono::steady_clock::now();
	std::cout << "TV denoising on GPU with CUDA, Time = " << std::chrono::duration_cast<std::chrono::microseconds>(session_end - session_begin).count() / 1000000.0 << " sec" << std::endl;


	//Save image to file
	cv::imwrite("D:\\Workspace\\TV-Denoise-PrimalDual-GPU-Build\\bin\\Release\\lenna_512_denoised_gpu.bmp", im_denoised);
	//--------------------------------------------------------------------------------------------------------



	//--------------------------------------------------------------------------------------------------------
	//Test CPU version
	cv::Mat im_denoised_cpu;
	session_begin = std::chrono::steady_clock::now();
	oTV->TV_Denoise_CPU(im_noisy, im_denoised_cpu, 0.1f, 1000);
	session_end = std::chrono::steady_clock::now();
	std::cout << "TV denoising on CPU, Time = " << std::chrono::duration_cast<std::chrono::microseconds>(session_end - session_begin).count() / 1000000.0 << " sec" << std::endl;

	//Save image to file
	cv::imwrite("D:\\Workspace\\TV-Denoise-PrimalDual-GPU-Build\\bin\\Release\\lenna_512_denoised_cpu.bmp", im_denoised_cpu);
	//--------------------------------------------------------------------------------------------------------
}


