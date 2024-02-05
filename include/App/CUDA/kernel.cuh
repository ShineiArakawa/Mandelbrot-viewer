#pragma once
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <iostream>

// CUDAのエラーチェック用のマクロ
#define CUDA_CHECK_ERROR(ans) \
  { cudaCheckError((ans), __FILE__, __LINE__); }

inline void cudaCheckError(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " at " << file << ":" << line << std::endl;
    exit(EXIT_FAILURE);
  }
}

__global__ void kernel(uchar4* devPtr, int width, int height, float tick);
void launchCUDAKernel(uchar4* bitmap, const int windowWidth, const int windowHeight, const float tick);