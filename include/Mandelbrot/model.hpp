#pragma once
#ifndef __INCLUDE_MANDELBROT_MODEL_HPP__

#define MY_PI 3.141592653589793238462643

#include <omp.h>

#include <Common/GUI.hpp>
#include <algorithm>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

namespace mandel {
namespace model {

class MandelbrotModel {
 private:
  std::shared_ptr<mandel::GUI::TextureBuffer> _textureBuffer = nullptr;

 public:
  MandelbrotModel();
  ~MandelbrotModel();

  inline static const double LOG2 = std::log(2);

  inline static const int TEXTURE_BUFFER_WIDTH = 1024;
  inline static const int TEXTURE_BUFFER_HEIGHT = 1024;
  double minX = -2.0;
  double maxX = 2.0;
  double minY = -2.0;
  double maxY = 2.0;
  int maxIter = 100;
  double threshold = 2.0;
  double alphaCoeff = 0.05;
  double density = 0.35;
  bool isEnabledSmoothing = false;
  bool isEnabledSinuidalColor = false;

  // Rendering
  void update();
  std::shared_ptr<mandel::GUI::TextureBuffer> getTexture() { return _textureBuffer; };
  ImTextureID getTextureID() { return (ImTextureID)_textureBuffer->getTextureID(); };

  // Mandelbrot calculation
  std::vector<double> arrange(const double, const double, const int);
  int isIncluded(std::complex<double>, const int, const double);
};
}  // namespace model
}  // namespace mandel

#define __INCLUDE_MANDELBROT_MODEL_HPP__
#endif