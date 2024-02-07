#pragma once

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stb_image.h>
#include <stb_image_write.h>

#include <Common/FileUtil.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace mandel {
namespace Image {
using Image_t = std::shared_ptr<std::vector<std::vector<std::vector<int>>>>;
using fs = mandel::fs::FileUtil;

unsigned char* loadTexture(const std::string& filePath, int& texWidth, int& texHeight);
void saveImage(const int width, const int height, const int channels, unsigned char* bytes, const std::string filePath);
Image_t fromBytes(const unsigned char* bytePixels, const int imageWidth, const int imageHeight, const int nChannels);
uint8_t* toBytes(Image_t pixels);

}  // namespace Image
}  // namespace mandel