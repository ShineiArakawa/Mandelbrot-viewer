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

#include <Common/CUDA.hpp>
#include <Common/FileUtil.hpp>
#include <Common/Image.hpp>
#include <iostream>
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
  cudaGraphicsResource* _deviceResource;

 public:
  TextureBuffer(const int, const int);
  TextureBuffer(std::string);
  TextureBuffer(const int, const int, unsigned char*);
  ~TextureBuffer();
  cudaGraphicsResource* getDeviceResource();
  unsigned int getTextureID();
  void initTexture(const int width, const int height, unsigned char* bytesTexture);
  void updateBuffer(unsigned char*);
  float getWidth() { return _width; };
  float getHeight() { return _height; };
};

}  // namespace GUI
}  // namespace mandel

#define __INCLUDE_GUI_HPP__
#endif