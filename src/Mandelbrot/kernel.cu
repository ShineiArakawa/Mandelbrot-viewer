#include <Mandelbrot/kernel.cuh>

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
    const bool isEnabledSuperSampling,
    const double density) {
  const int indexX = threadIdx.x + blockIdx.x * blockDim.x;
  const int indexY = threadIdx.y + blockIdx.y * blockDim.y;
  const int offset = indexX + indexY * width;
  const double thresholdSquared = threshold * threshold;

  if (indexX < width && indexY < height) {
    const double dX = (maxX - minX) / (double)(width - 1);
    const double dY = (maxY - minY) / (double)(height - 1);
    const double centerCoordX = (double)indexX * dX + minX;
    const double centerCoordY = (double)indexY * dY + minY;

    int stoppedIter = 0;
    double radius = 0.0;
    if (isEnabledSuperSampling) {
      const double offsetX = dX / 4.0;
      const double offsetY = dY / 4.0;

      for (int superSampleX = 0; superSampleX < 2; superSampleX++) {
        for (int superSampleY = 0; superSampleY < 2; superSampleY++) {
          const double coordX = (double)superSampleX * 2.0 * offsetX + centerCoordX - offsetX;
          const double coordY = (double)superSampleY * 2.0 * offsetY + centerCoordY - offsetY;

          double iRadiusSquared = 0.0;
          int iStoppedIter = 0;
          double zReal = 0.0;
          double zImag = 0.0;

          for (int iter = 0; iter < maxIter; iter++) {
            const double zRealSquared = zReal * zReal;
            const double zImagSquared = zImag * zImag;
            iRadiusSquared = zRealSquared + zImagSquared;

            if (iRadiusSquared > thresholdSquared) {
              iStoppedIter = iter;
              break;
            }

            const double tmp_zReal = zRealSquared - zImagSquared + coordX;
            const double tmp_zImag = 2.0 * zReal * zImag + coordY;
            zReal = tmp_zReal;
            zImag = tmp_zImag;
          }

          radius += sqrt(iRadiusSquared);
          stoppedIter += iStoppedIter;
        }
      }

      radius /= 4.0;
      stoppedIter /= 4;
    } else {
      double radiusSquared = 0.0;
      double zReal = 0.0;
      double zImag = 0.0;

      for (int iter = 0; iter < maxIter; iter++) {
        const double zRealSquared = zReal * zReal;
        const double zImagSquared = zImag * zImag;
        radiusSquared = zRealSquared + zImagSquared;

        if (radiusSquared > thresholdSquared) {
          stoppedIter = iter;
          break;
        }

        const double tmp_zReal = zRealSquared - zImagSquared + centerCoordX;
        const double tmp_zImag = 2.0 * zReal * zImag + centerCoordY;
        zReal = tmp_zReal;
        zImag = tmp_zImag;
      }

      radius = sqrt(radiusSquared);
    }

    int R, G, B, A = 0;
    double alpha = 0.0;

    if (radius > threshold) {
      if (isEnabledSmoothing) {
        // Smoothing
        const double log2Val = log(2.0);
        const double nu = log(log(radius) / log2Val) / log2Val;
        alpha = alphaCoeff * ((double)stoppedIter + 1.0 - nu);
      } else {
        alpha = alphaCoeff * (double)stoppedIter;
      }

      if (isEnabledSinuidalColor) {
        // Color grading
        alpha = alpha * density;
        alpha = log(alpha + 1.0);

        const double factorR = (cos((alpha * 2.0 - 1.0) * M_PI) + 1.0) * 0.5;
        const double factorG = (cos((alpha * 2.0 - 0.75) * M_PI) + 1.0) * 0.5;
        const double factorB = (cos((alpha * 2.0 - 0.5) * M_PI) + 1.0) * 0.5;

        R = (int)(factorR * 255.0);
        G = (int)(factorG * 255.0);
        B = (int)(factorB * 255.0);
        A = 255;
      } else {
        alpha = max(min(alpha, 1.0), 0.0);
        const int pixelValue = max(min((int)(alpha * 255.0), 255), 0);
        R = pixelValue;
        G = pixelValue;
        B = pixelValue;
        A = 255;
      }
    }

    deviceArray[offset].x = R;
    deviceArray[offset].y = G;
    deviceArray[offset].z = B;
    deviceArray[offset].w = A;
  }
}

void launchCUDAKernel(
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
    const bool isEnabledSuperSampling,
    const double density) {
  dim3 blockDim(16, 16);
  dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

  kernel<<<gridDim, blockDim>>>(
      deviceArray,
      minX,
      minY,
      maxX,
      maxY,
      width,
      height,
      maxIter,
      threshold,
      isEnabledSmoothing,
      alphaCoeff,
      isEnabledSinuidalColor,
      isEnabledSuperSampling,
      density);
}
