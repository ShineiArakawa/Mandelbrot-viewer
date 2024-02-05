#include <App/CUDA/kernel.cuh>

// CUDAカーネル：単純な色のパターンを生成
__global__ void kernel(uchar4* devPtr, int width, int height, float tick) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  if (x < width && y < height) {
  float theta = 10.0f * tick / 60.0f * 2.0f * M_PI;
  float theta_x = x / 60.0f * 2.0f * M_PI;
  float theta_y = y / 60.0f * 2.0f * M_PI;
  float r = fabs(sin(theta + theta_x));
  float g = fabs(cos(theta + theta_y));
  float b = fabs(sin(theta + theta_x) * cos(theta + theta_y));

  devPtr[offset].x = (unsigned char)(r * 255);
  devPtr[offset].y = (unsigned char)(g * 255);
  devPtr[offset].z = (unsigned char)(b * 255);
  devPtr[offset].w = 255;
  }
}

void launchCUDAKernel(uchar4* d_data, const int windowWidth, const int windowHeight, const float time) {
  dim3 blockDim(16, 16);
  dim3 gridDim((windowWidth + blockDim.x - 1) / blockDim.x, (windowHeight + blockDim.y - 1) / blockDim.y);
  kernel<<<gridDim, blockDim>>>(d_data, windowWidth, windowHeight, time);
}
