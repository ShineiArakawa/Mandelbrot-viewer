#include <Mandelbrot/model.hpp>

namespace mandel {
namespace model {
MandelbrotModel::MandelbrotModel(bool offlineMode_) {
  offlineMode = offlineMode_;
  if (!offlineMode) {
    _textureBuffer = std::make_shared<mandel::GUI::TextureBuffer>(TEXTURE_BUFFER_WIDTH, TEXTURE_BUFFER_HEIGHT);
  }
}

MandelbrotModel::~MandelbrotModel() {
}

void MandelbrotModel::update() {
  if (isEnabledDollyOut) {
    const double deltaX = (maxX - minX) * dollyOutFactor;
    const double deltaY = (maxY - minY) * dollyOutFactor;
    minX = minX - deltaX;
    minY = minY - deltaY;
    maxX = maxX + deltaX;
    maxY = maxY + deltaY;

    if (minX > maxX) {
      minX = maxX;
    }
    if (minY > maxY) {
      minY = maxY;
    }
  }

  cudaGraphicsResource* deviceResource = _textureBuffer->getDeviceResource();

  cudaArray* deviceArray;
  uchar4* cudaBitmap;
  {
    // Turn on mapped OpenGL texture
    CUDA_CHECK_ERROR(cudaGraphicsMapResources(1, &deviceResource, nullptr));
    // Get data as array
    CUDA_CHECK_ERROR(cudaGraphicsSubResourceGetMappedArray(&deviceArray, deviceResource, 0, 0));
    // Malloc cuda memory
    CUDA_CHECK_ERROR(cudaMalloc((void**)&cudaBitmap, TEXTURE_BUFFER_WIDTH * TEXTURE_BUFFER_HEIGHT * sizeof(uchar4)));
  }
  {
    // Launch the kernel and write data to cuda memory
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
        isVisibleAIM,
        density);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
  }
  {
    // Copy cuda memory to mapped OpenGL texture
    CUDA_CHECK_ERROR(cudaMemcpy2DToArray(deviceArray, 0, 0, cudaBitmap, TEXTURE_BUFFER_WIDTH * sizeof(uchar4), TEXTURE_BUFFER_WIDTH * sizeof(uchar4), TEXTURE_BUFFER_HEIGHT, cudaMemcpyDeviceToDevice));
    // Discard cuda memory
    CUDA_CHECK_ERROR(cudaFree(cudaBitmap));
    // Turn off mapped OpenGL texture
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

void MandelbrotModel::saveCurrentTexture(const std::string filePath) {
  unsigned char* bytesTexture = (unsigned char*)malloc(sizeof(unsigned char) * TEXTURE_BUFFER_WIDTH * TEXTURE_BUFFER_HEIGHT * 4);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, bytesTexture);
  mandel::Image::saveImage(TEXTURE_BUFFER_WIDTH, TEXTURE_BUFFER_HEIGHT, 4, bytesTexture, filePath);
}

}  // namespace model
}  // namespace mandel