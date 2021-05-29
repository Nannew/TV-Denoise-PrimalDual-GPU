# TV-L1 denoising with Primal-Dual using CUDA

This repository gives an example of how to implement an image processing algorithm on the GPU with Nvidia CUDA and CMake.
1. The example compared the implementation of the same TV-L1 denoising algorithm on the CPU and GPU.
2. The OpenCV CUDA module (cv::cuda::GpuMat ) is used to host the image data instead of the common method. This requires building OpenCV with CUDA.

# Experiment Result

The test image is a standard 512x512 Grayscale (8-bit) Lena image from [Standard Test Images](https://www.ece.rice.edu/~wakin/images/). A zero-mean Gaussian white noise with an intensity-dependent variance of 0.01 is added to make a noisy image for the test.

The C++ CPU implementation needs 4.13699 seconds to run 1000 iterations on a 512x512 gray-scale image during the experiment, 0.0041 second per iteration. While the C++ GPU implementation took only 0.271657 seconds to run 1000 iterations, 0.00027 second per iteration. Therefore, for an image size of 512x512, parallel the code with GPU is more than 15 times faster than CPU implementation. This difference will become greater as the image size increases.

Note that for the GPU implementation, the data transfer time between CPU and GPU is also considered during the comparison. The GPU implementation needs to upload the image from CPU to GPU before computation and download image back to CPU after convergence. However, this time is trivial compared with the total computation time.

<img src="../master/Data/Test_Result.png?raw=true" width="700" >

* Original 512x512 Grayscale (8-bit) Lena image (Left) and noisy image after adding Gaussian noise (Right). 

<img src="../master/Data/lena512.bmp?raw=true" width="400" height="400">  <img src="../master/Data/lenna_512_noisy.bmp?raw=true" width="400" height="400">


* Denoised image using TV-L1 denoising on GPU (Left) and CPU (Right). The denoised image after processing is very close after 1000 iterations. 

<img src="../master/Data/lenna_512_denoised_gpu.bmp?raw=true" width="400" height="400">  <img src="../master/Data/lenna_512_denoised_cpu.bmp?raw=true" width="400" height="400">