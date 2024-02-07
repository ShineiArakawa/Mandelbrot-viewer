#pragma once
#ifndef __INCLUDE_MANDELBROT_MODEL_HPP__

#define MY_PI 3.141592653589793238462643

#include <omp.h>

#include <Common/GUI.hpp>
#include <Mandelbrot/kernel.cuh>
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
  MandelbrotModel(bool offlineMode_ = false);
  ~MandelbrotModel();

  inline static const double LOG2 = std::log(2);

  inline static const int TEXTURE_BUFFER_WIDTH = 1024;
  inline static const int TEXTURE_BUFFER_HEIGHT = 1024;
  double minX = -2.0;
  double maxX = 2.0;
  double minY = -2.0;
  double maxY = 2.0;
  int maxIter = 1000;
  double threshold = 5.0;
  double alphaCoeff = 0.05;
  double density = 0.35;
  bool isEnabledSmoothing = false;
  bool isEnabledSinuidalColor = false;
  bool isEnabledSuperSampling = false;
  double dollyOutFactor = 0.001;
  bool isEnabledDollyOut = false;
  int superSampleFactor = 1;
  bool offlineMode = false;
  unsigned char* bytePixelsBuffer = (unsigned char*)calloc(sizeof(unsigned char), TEXTURE_BUFFER_WIDTH* TEXTURE_BUFFER_HEIGHT * 4);

  // Rendering
  void update();
  std::shared_ptr<mandel::GUI::TextureBuffer> getTexture() { return _textureBuffer; };
  ImTextureID getTextureID() { return (ImTextureID)_textureBuffer->getTextureID(); };

  // Mandelbrot calculation
  std::vector<double> arrange(const double, const double, const int);

  void saveCurrentTexture(const std::string);
};
}  // namespace model
}  // namespace mandel

#define __INCLUDE_MANDELBROT_MODEL_HPP__
#endif