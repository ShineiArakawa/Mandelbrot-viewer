#include <Common/GUI.hpp>
#include <iostream>

namespace mandel {
namespace GUI {

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

}  // namespace GUI
}  // namespace mandel