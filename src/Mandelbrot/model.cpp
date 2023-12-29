#include <Mandelbrot/model.hpp>

namespace mandel {
namespace model {
static int counter = 0;
MandelbrotModel::MandelbrotModel() {
  _textureBuffer = std::make_shared<mandel::GUI::TextureBuffer>(TEXTURE_BUFFER_WIDTH, TEXTURE_BUFFER_HEIGHT);
}

MandelbrotModel::~MandelbrotModel() {
}

void MandelbrotModel::update() {
  auto xCoords = arrange(minX, maxX, TEXTURE_BUFFER_WIDTH);
  auto yCoords = arrange(minY, maxY, TEXTURE_BUFFER_HEIGHT);
  auto bytePixels = (unsigned char*)calloc(sizeof(unsigned char), TEXTURE_BUFFER_WIDTH * TEXTURE_BUFFER_HEIGHT * 4);

#pragma omp parallel for
  for (int x = 0; x < (int)TEXTURE_BUFFER_WIDTH; x++) {
#pragma omp parallel for
    for (int y = 0; y < (int)TEXTURE_BUFFER_HEIGHT; y++) {
      std::complex<double> coord(xCoords[x], yCoords[y]);
      std::complex<double> z(0.0, 0.0);

      int stoppedIter = 0;
      double radius = 0.0;

      for (int iter = 0; iter < maxIter; iter++) {
        radius = std::abs(z);

        if (radius > threshold) {
          stoppedIter = iter;
          break;
        }

        z = z * z + coord;
      }

      if (stoppedIter < maxIter) {
        double alpha = 0.0;
        const int index = (x * TEXTURE_BUFFER_HEIGHT + y) * 4;

        if (isEnabledSmoothing) {
          const double nu = std::log(std::log(radius) / LOG2) / LOG2;
          alpha = alphaCoeff * ((double)stoppedIter + 1.0 - nu);
        } else {
          alpha = alphaCoeff * (double)stoppedIter;
        }

        int R, G, B, A;

        if (isEnabledSinuidalColor) {
          alpha = alpha * density;
          alpha = std::log(alpha + 1.0);

          double factorR = (std::cos((alpha * 2.0 - 1.0) * MY_PI) + 1.0) * 0.5;
          double factorG = (std::cos((alpha * 2.0 - 0.75) * MY_PI) + 1.0) * 0.5;
          double factorB = (std::cos((alpha * 2.0 - 0.5) * MY_PI) + 1.0) * 0.5;

          R = (int)(factorR * 255.0);
          G = (int)(factorG * 255.0);
          B = (int)(factorB * 255.0);
          A = 255;
        } else {
          const int pixelValue = std::max(std::min((int)(alpha * 255.0), 255), 0);
          R = pixelValue;
          G = pixelValue;
          B = pixelValue;
          A = 255;
        }

        bytePixels[index] = R;
        bytePixels[index + 1] = G;
        bytePixels[index + 2] = B;
        bytePixels[index + 3] = A;
      }
    }
  }

  _textureBuffer->updateBuffer(bytePixels);
  std::cout << "Update !" << std::endl;
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

int MandelbrotModel::isIncluded(std::complex<double> coord, const int maxIter, const double threshold) {
  std::complex<double> z(0.0, 0.0);
  for (int i = 0; i < maxIter; i++) {
    if (std::abs(z) > threshold) {
      return i;
    }

    z = z * z + coord;
  }

  return -1;
}

}  // namespace model
}  // namespace mandel