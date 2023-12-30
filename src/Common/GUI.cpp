#include <Common/FIleUtil.hpp>
#include <Common/GUI.hpp>
#include <iostream>

namespace mandel {
namespace GUI {

using fs = mandel::fs::FileUtil;

TextureBuffer::TextureBuffer(const int width, const int height) {
  initTexture(width, height, nullptr);
};

TextureBuffer::TextureBuffer(const int width, const int height, unsigned char* bytesTexture) {
  initTexture(width, height, bytesTexture);
}

TextureBuffer::TextureBuffer(std::string filePath) {
  int width, height;
  unsigned char* bytesTexture = loadTexture(filePath, width, height);

  initTexture(width, height, bytesTexture);
}

void TextureBuffer::initTexture(const int width, const int height, unsigned char* bytesTexture) {
  glGenTextures(1, &_texture);
  glBindTexture(GL_TEXTURE_2D, _texture);

  // Setup filtering parameters for display
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);  // This is required on WebGL for non power-of-two textures
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);  // Same

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, bytesTexture);

  stbi_image_free(bytesTexture);

  _width = (float)width;
  _height = (float)height;
}

TextureBuffer::~TextureBuffer() {
  glDeleteTextures(1, &_texture);
}

void TextureBuffer::updateBuffer(unsigned char* bytePixels) {
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, bytePixels);
}

unsigned int TextureBuffer::getTextureID() {
  return _texture;
}

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
  const std::string dirPath = fs::dirPath(fs::absPath(filePath));
  if (!fs::exists(dirPath)) {
    fs::mkdirs(dirPath);
  }

  const std::string extension = fs::extension(filePath);
  if (extension == ".png") {
    stbi_write_png(filePath.c_str(), width, height, channels, bytes, 0);
  } else if (extension == ".jpg") {
    stbi_write_jpg(filePath.c_str(), width, height, channels, bytes, 100);
  } else {
    fprintf(stderr, "Unsupported save image format: %s\n", extension.c_str());
  }
};

}  // namespace GUI
}  // namespace mandel