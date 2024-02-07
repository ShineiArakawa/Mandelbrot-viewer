#include <Common/GUI.hpp>

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
  unsigned char* bytesTexture = mandel::Image::loadTexture(filePath, width, height);

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
  CUDA_CHECK_ERROR(cudaGraphicsGLRegisterImage(&_deviceResource, _texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

  _width = (float)width;
  _height = (float)height;
}

TextureBuffer::~TextureBuffer() {
  glDeleteTextures(1, &_texture);
}

void TextureBuffer::updateBuffer(unsigned char* bytePixels) {
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, bytePixels);
}

cudaGraphicsResource* TextureBuffer::getDeviceResource() {
  return _deviceResource;
}

unsigned int TextureBuffer::getTextureID() {
  return _texture;
}

}  // namespace GUI
}  // namespace mandel