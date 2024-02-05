#pragma once

#include <Common/CUDA.hpp>
#include <iostream>

__global__ void kernel(
    uchar4* deviceArray,
    const double minX,
    const double minY,
    const double maxX,
    const double maxY,
    const int width,
    const int height,
    const int maxIter,
    const double threshold,
    const bool isEnabledSmoothing,
    const double alphaCoeff,
    const bool isEnabledSinuidalColor,
    const double density);
void launchCUDAKernel(uchar4* deviceArray,
    const double minX,
    const double minY,
    const double maxX,
    const double maxY,
    const int width,
    const int height,
    const int maxIter,
    const double threshold,
    const bool isEnabledSmoothing,
    const double alphaCoeff,
    const bool isEnabledSinuidalColor,
    const double density);