#pragma once
#ifndef __INCLUDE_CUDA_MAIN_HPP__

#include <argparse/argparse.hpp>
#include <iostream>
#include <memory>
#include <string>

inline static const int WINDOW_WIDTH = 1200;
inline static const int WINDOW_HEIGHT = 800;
inline static const char* WINDOW_TITLE = "Mandelbrot";
inline static const int SETTING_TAB_HEIGHT = 300;
inline static const int SIDEBAR_WIDTH = 400;
inline static const float MOUSE_WHEEL_THRESHOLD = 0.01;
inline static const float MOUSE_WHEEL_DELTA = 0.1;

int offlineRender(argparse::ArgumentParser&);
int launchWindow();

#define __INCLUDE_CUDA_MAIN_HPP__
#endif