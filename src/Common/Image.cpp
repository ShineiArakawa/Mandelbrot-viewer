#include <Common/Image.hpp>

namespace mandel {
namespace Image {

unsigned char* loadTexture(const std::string& filePath, int& texWidth, int& texHeight) {
  int channels;

  // Texture
  unsigned char* bytesTexture = stbi_load(filePath.c_str(), &texWidth, &texHeight, &channels, STBI_rgb_alpha);
  if (!bytesTexture) {
    fprintf(stderr, "Failed to load image file: %s\n", filePath.c_str());
    exit(1);
  }

  return bytesTexture;
};

void saveImage(const int width, const int height, const int channels, unsigned char* bytes, const std::string filePath) {
  const std::string dirPath = fs::dirPath(filePath);
  if (!fs::exists(dirPath)) {
    fs::mkdirs(dirPath);
  }
  std::cout << "filePath=" << filePath << std::endl;
  const std::string extension = fs::extension(filePath);
  if (extension == ".png") {
    stbi_write_png(filePath.c_str(), width, height, channels, bytes, 0);
  } else if (extension == ".jpg") {
    stbi_write_jpg(filePath.c_str(), width, height, channels, bytes, 100);
  } else {
    fprintf(stderr, "Unsupported save image format: %s\n", extension.c_str());
  }
};

Image_t fromBytes(const unsigned char* bytePixels, const int imageWidth, const int imageHeight, const int nChannels) {
  // Alloc
  Image_t pixels = std::make_shared<std::vector<std::vector<std::vector<int>>>>();

  for (int w = 0; w < imageWidth; w++) {
    pixels->push_back(std::vector<std::vector<int>>());

    for (int h = 0; h < imageHeight; h++) {
      (*pixels)[w].push_back(std::vector<int>());
    }
  }

  // Convert
  for (int w = 0; w < imageWidth; w++) {
    for (int h = 0; h < imageHeight; h++) {
      const unsigned char* texel = bytePixels + (w + imageWidth * h) * nChannels;

      for (int channel = 0; channel < nChannels; channel++) {
        const unsigned char charValue = texel[channel];
        const int value = charValue;
        (*pixels)[w][h].push_back(value);
      }
    }
  }

  return std::move(pixels);
};

uint8_t* toBytes(Image_t pixels) {
  uint8_t* bytePixels = nullptr;

  if (pixels != nullptr) {
    int width = pixels->size();
    int height = (*pixels)[0].size();
    int nChannels = (*pixels)[0][0].size();

    bytePixels = (uint8_t*)malloc(width * height * nChannels);

    int index = 0;
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int channel = 0; channel < nChannels; channel++) {
          const int value = (*pixels)[w][h][channel];
          bytePixels[index++] = value;
        }
      }
    }
  }

  return bytePixels;
};

}  // namespace Image
}  // namespace mandel