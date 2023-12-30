#pragma once
#ifndef __INCLUDE_GUI_HPP__

#define GLAD_GL_IMPLEMENTATION
#include <glad/glad.h>

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

// clang-format off
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_glut.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui.h>
#include <imgui_internal.h>
// clang-format on

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stb_image.h>
#include <stb_image_write.h>

#include <memory>
#include <string>
#include <vector>

namespace mandel {
namespace GUI {

class TextureBuffer {
 private:
  float _width;
  float _height;

  GLuint _texture;

 public:
  TextureBuffer(const int, const int);
  TextureBuffer(std::string);
  TextureBuffer(const int, const int, unsigned char*);
  ~TextureBuffer();
  unsigned int getTextureID();
  void initTexture(const int width, const int height, unsigned char* bytesTexture);
  void updateBuffer(unsigned char*);
  float getWidth() { return _width; };
  float getHeight() { return _height; };
};

unsigned char* loadTexture(const std::string& filePath, int& texWidth, int& texHeight);
void saveImage(const int width, const int height, const int channels, unsigned char* bytes, const std::string filePath);

}  // namespace GUI
}  // namespace mandel

#define __INCLUDE_GUI_HPP__
#endif