#include <Mandelbrot/model.hpp>

namespace mandel {
namespace model {
static int counter = 0;
MandelbrotModel::MandelbrotModel(bool offlineMode_) {
  offlineMode = offlineMode_;
  if (!offlineMode) {
    _textureBuffer = std::make_shared<mandel::GUI::TextureBuffer>(TEXTURE_BUFFER_WIDTH, TEXTURE_BUFFER_HEIGHT);
  }
}

MandelbrotModel::~MandelbrotModel() {
}

void MandelbrotModel::update() {
  // auto xCoords = arrange(minX, maxX, TEXTURE_BUFFER_WIDTH);
  // auto yCoords = arrange(minY, maxY, TEXTURE_BUFFER_HEIGHT);

  // #pragma omp parallel for
  //   for (int x = 0; x < (int)TEXTURE_BUFFER_WIDTH; x++) {
  // #pragma omp parallel for
  //     for (int y = 0; y < (int)TEXTURE_BUFFER_HEIGHT; y++) {
  //       const std::complex<double> coord(xCoords[x], yCoords[y]);
  //       std::complex<double> z(0.0, 0.0);

  //       int stoppedIter = 0;
  //       double radius = 0.0;

  //       for (int iter = 0; iter < maxIter; iter++) {
  //         radius = std::abs(z);

  //         if (radius > threshold) {
  //           stoppedIter = iter;
  //           break;
  //         }

  //         z = z * z + coord;
  //       }

  //       if (stoppedIter < maxIter) {
  //         double alpha = 0.0;
  //         const int index = (x * TEXTURE_BUFFER_HEIGHT + y) * 4;

  //         if (isEnabledSmoothing) {
  //           const double nu = std::log(std::log(radius) / LOG2) / LOG2;
  //           alpha = alphaCoeff * ((double)stoppedIter + 1.0 - nu);
  //         } else {
  //           alpha = alphaCoeff * (double)stoppedIter;
  //         }

  //         int R, G, B, A;

  //         if (isEnabledSinuidalColor) {
  //           alpha = alpha * density;
  //           alpha = std::log(alpha + 1.0);

  //           const double factorR = (std::cos((alpha * 2.0 - 1.0) * MY_PI) + 1.0) * 0.5;
  //           const double factorG = (std::cos((alpha * 2.0 - 0.75) * MY_PI) + 1.0) * 0.5;
  //           const double factorB = (std::cos((alpha * 2.0 - 0.5) * MY_PI) + 1.0) * 0.5;

  //           R = (int)(factorR * 255.0);
  //           G = (int)(factorG * 255.0);
  //           B = (int)(factorB * 255.0);
  //           A = 255;
  //         } else {
  //           const int pixelValue = std::max(std::min((int)(alpha * 255.0), 255), 0);
  //           R = pixelValue;
  //           G = pixelValue;
  //           B = pixelValue;
  //           A = 255;
  //         }

  //         bytePixelsBuffer[index] = R;
  //         bytePixelsBuffer[index + 1] = G;
  //         bytePixelsBuffer[index + 2] = B;
  //         bytePixelsBuffer[index + 3] = A;
  //       }
  //     }
  //   }

  cudaGraphicsResource* deviceResource = _textureBuffer->getDeviceResource();

  cudaArray* deviceArray;
  uchar4* cudaBitmap;
  {
    CUDA_CHECK_ERROR(cudaGraphicsMapResources(1, &deviceResource, nullptr));

    CUDA_CHECK_ERROR(cudaGraphicsSubResourceGetMappedArray(&deviceArray, deviceResource, 0, 0));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&cudaBitmap, TEXTURE_BUFFER_WIDTH * TEXTURE_BUFFER_HEIGHT * sizeof(uchar4)));
  }
  {
    launchCUDAKernel(
        cudaBitmap,
        minY,
        minX,
        maxY,
        maxX,
        TEXTURE_BUFFER_WIDTH,
        TEXTURE_BUFFER_HEIGHT,
        maxIter,
        threshold,
        isEnabledSmoothing,
        alphaCoeff,
        isEnabledSinuidalColor,
        isEnabledSuperSampling,
        density);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
  }
  {
    CUDA_CHECK_ERROR(cudaMemcpy2DToArray(deviceArray, 0, 0, cudaBitmap, TEXTURE_BUFFER_WIDTH * sizeof(uchar4), TEXTURE_BUFFER_WIDTH * sizeof(uchar4), TEXTURE_BUFFER_HEIGHT, cudaMemcpyDeviceToDevice));
    CUDA_CHECK_ERROR(cudaFree(cudaBitmap));
    CUDA_CHECK_ERROR(cudaGraphicsUnmapResources(1, &deviceResource, nullptr));
  }
}

std::vector<double> MandelbrotModel::arrange(const double min, const double max, const int length) {
  std::vector<double> array;
  const double stride = (max - min) / (double)(length - 1);
  for (int i = 0; i < length; i++) {
    const double value = min + stride * (double)i;
    array.push_back(value);
  }
  return array;
}

}  // namespace model
}  // namespace mandel