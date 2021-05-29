#include "TVDenoiseGPU.h"


TVDenoiseGPU::TVDenoiseGPU()
{
	
}

TVDenoiseGPU::~TVDenoiseGPU()
{

}



__global__ void PxPy_kernel(
	cv::cuda::PtrStepSz<float> X,
	cv::cuda::PtrStepSz<float> Px,
	cv::cuda::PtrStepSz<float> Py,
	float currsigma,
	int rows,
	int cols
)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;//rows
	int j = blockIdx.x * blockDim.x + threadIdx.x;//columns

	int i_next = min(i+1, rows-1);
	int j_next = min(j+1, cols-1);


	float dx, dy, m;


	dx = (X.ptr(i)[j_next] - X.ptr(i)[j])*currsigma + Px.ptr(i)[j];
	dy = (X.ptr(i_next)[j] - X.ptr(i)[j])*currsigma + Py.ptr(i)[j];
	m = 1.0 / max(sqrt(dx*dx + dy * dy), 1.0f);
	Px.ptr(i)[j] = dx * m;
	Py.ptr(i)[j] = dy * m;
};

__global__ void DeltaX_kernel(
	cv::cuda::PtrStepSz<float> X,
	cv::cuda::PtrStepSz<float> Px,
	cv::cuda::PtrStepSz<float> Py,
	int rows,
	int cols
)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;//rows
	int j = blockIdx.x * blockDim.x + threadIdx.x;//columns

	int i_next = min(i + 1, rows - 1);
	int j_next = min(j + 1, cols - 1);


	float dx, dy, m;


	Px.ptr(i)[j] = (X.ptr(i)[j_next] - X.ptr(i)[j]);
	Py.ptr(i)[j] = (X.ptr(i_next)[j] - X.ptr(i)[j]);
};


__global__ void Update_X_kernel2(
	cv::cuda::PtrStepSz<float> observations,
	cv::cuda::PtrStepSz<float> X,
	cv::cuda::PtrStepSz<float> Px,
	cv::cuda::PtrStepSz<float> Py,
	float clambda,
	float sigma,
	float tau,
	float theta,
	float lt,
	int rows,
	int cols
)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;//rows
	int j = blockIdx.x * blockDim.x + threadIdx.x;//columns

	int i_prev = max(i - 1, 0);
	int j_prev = max(j - 1, 0);

	/// X1 = X + tau*(-nablaT(P))
	float v = X.ptr(i)[j] + tau * (Px.ptr(i)[j] - Px.ptr(i)[j_prev] + Py.ptr(i)[j] - Py.ptr(i_prev)[j]);
	float v_minus_lt = v - observations.ptr(i)[j];
	float x_new;
	if (v_minus_lt > lt)
	{
		x_new = v - lt;
	}
	else if (v_minus_lt < -lt)
	{
		x_new = v + lt;
	}
	else
	{
		x_new = observations.ptr(i)[j];
	}

	/// X = X2 + theta*(X2 - X)
	X.ptr(i)[j] = x_new + theta * (x_new - X.ptr(i)[j]);
};



//But actually the speed is similar to clone gpuMat directly
__global__ void Clone_GpuMat_kernel(
	cv::cuda::PtrStepSz<float> observations,
	cv::cuda::PtrStepSz<float> X)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;//rows
	int j = blockIdx.x * blockDim.x + threadIdx.x;//columns

	X.ptr(i)[j] = observations.ptr(i)[j];
};






void TVDenoiseGPU::TV_Denoise_GPU(const cv::cuda::GpuMat& observations, cv::cuda::GpuMat& X, float clambda, int niters)
{
	//CV_Assert(observations.size() > 0 && niters > 0 && lambda > 0);

	const float L2 = 8.0, tau = 0.02, sigma = 1. / (L2*tau), theta = 1.0, lt = clambda * tau;
	//float clambda = (float)lambda;
	//float s = 0;
	//const int workdepth = CV_32F;

	int i, x, y, rows = observations.rows, cols = observations.cols;



	const dim3 block(16, 16);
	// Calculate grid size to cover the whole image
	const dim3 grid(cv::cuda::device::divUp(cols, block.x), cv::cuda::device::divUp(rows, block.y));



	///X = observations.clone();
	X.create(rows, cols, CV_32FC1);
	Clone_GpuMat_kernel<<<grid, block>>>(
		static_cast<cv::cuda::PtrStepSz<float>>(observations),
		static_cast<cv::cuda::PtrStepSz<float>>(X));
	
	//cv::cuda::GpuMat Px(rows, cols, CV_32FC1, 0.0f);
	//cv::cuda::GpuMat Py(rows, cols, CV_32FC1, 0.0f);
	cv::cuda::GpuMat Px(rows, cols, CV_32FC1);
	cv::cuda::GpuMat Py(rows, cols, CV_32FC1);
	DeltaX_kernel <<<grid, block >>> (
		static_cast<cv::cuda::PtrStepSz<float>>(X),
		static_cast<cv::cuda::PtrStepSz<float>>(Px),
		static_cast<cv::cuda::PtrStepSz<float>>(Py),
		rows,
		cols
		);

	//cv::cuda::GpuMat Rs(rows, cols, CV_32FC1, 0.0f);

	

	for (i = 0; i < niters; i++)
	{
		float currsigma = i == 0 ? 1 + sigma : sigma;

		///////////////////////////////////////////////////////////////////////
		// P_ = P + sigma*nabla(X)
		// P(x,y) = P_(x,y)/max(||P(x,y)||,1)

		PxPy_kernel<<<grid, block>>>(
			static_cast<cv::cuda::PtrStepSz<float>>(X),
			static_cast<cv::cuda::PtrStepSz<float>>(Px),
			static_cast<cv::cuda::PtrStepSz<float>>(Py),
			currsigma,
			rows,
			cols
			);
		///////////////////////////////////////////////////////////////////////


		Update_X_kernel2<<<grid, block>>>(
			static_cast<cv::cuda::PtrStepSz<float>>(observations),
			static_cast<cv::cuda::PtrStepSz<float>>(X),
			static_cast<cv::cuda::PtrStepSz<float>>(Px),
			static_cast<cv::cuda::PtrStepSz<float>>(Py),
			clambda,
			sigma,
			tau,
			theta,
			lt,
			rows,
			cols
			);
	}
}





void TVDenoiseGPU::TV_Denoise_CPU(cv::Mat& observations, cv::Mat& X, float clambda, int niters)
{

	const float L2 = 8.0, tau = 0.02, sigma = 1. / (L2*tau), theta = 1.0, lt = clambda * tau;
	const int workdepth = CV_32F;

	int i, x, y, rows = observations.rows, cols = observations.cols;

	X = observations.clone();
	cv::Mat P = cv::Mat::zeros(rows, cols, CV_MAKETYPE(workdepth, 2));
	cv::Mat Rs = cv::Mat::zeros(rows, cols, workdepth);

	for (i = 0; i < niters; i++)
	{
		float currsigma = i == 0 ? 1 + sigma : sigma;
		for (y = 0; y < rows; y++)
		{
			const float* x_curr = X.ptr<float>(y);
			const float* x_next = X.ptr<float>(std::min(y + 1, rows - 1));
			cv::Point2f* p_curr = P.ptr<cv::Point2f>(y);
			float dx, dy, m;
			for (x = 0; x < cols - 1; x++)
			{
				dx = (x_curr[x + 1] - x_curr[x])*currsigma + p_curr[x].x;
				dy = (x_next[x] - x_curr[x])*currsigma + p_curr[x].y;
				m = 1.0 / std::max(std::sqrt(dx*dx + dy * dy), 1.0f);
				p_curr[x].x = dx * m;
				p_curr[x].y = dy * m;
			}
			dy = (x_next[x] - x_curr[x])*currsigma + p_curr[x].y;
			m = 1.0 / std::max(std::abs(dy), 1.0f);
			p_curr[x].x = 0.0;
			p_curr[x].y = dy * m;
		}

		float v, x_new, v_minus_lt;
		for (y = 0; y < rows; y++)
		{
			float* x_curr = X.ptr<float>(y);
			const cv::Point2f* p_curr = P.ptr<cv::Point2f>(y);
			const cv::Point2f* p_prev = P.ptr<cv::Point2f>(std::max(y - 1, 0));


			float* obs_curr = observations.ptr<float>(y);

			x = 0;
			v = x_curr[x] + tau * (p_curr[x].y - p_prev[x].y);
			v_minus_lt = v - obs_curr[x];
			if (v_minus_lt > lt)
			{
				x_new = v - lt;
			}
			else if (v_minus_lt < -lt)
			{
				x_new = v + lt;
			}
			else
			{
				x_new = obs_curr[x];
			}

			x_curr[x] = x_new + theta * (x_new - x_curr[x]);


			for (x = 1; x < cols; x++)
			{
				v = x_curr[x] + tau * (p_curr[x].x - p_curr[x - 1].x + p_curr[x].y - p_prev[x].y);
				v_minus_lt = v - obs_curr[x];
				if (v_minus_lt > lt)
				{
					x_new = v - lt;
				}
				else if (v_minus_lt < -lt)
				{
					x_new = v + lt;
				}
				else
				{
					x_new = obs_curr[x];
				}

				x_curr[x] = x_new + theta * (x_new - x_curr[x]);
			}
		}
	}

}