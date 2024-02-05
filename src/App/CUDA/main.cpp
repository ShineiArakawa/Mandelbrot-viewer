#define GLAD_GL_IMPLEMENTATION
#include <glad/glad.h>

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

#include <App/CUDA/kernel.cuh>

cudaGraphicsResource* deviceResource;
GLuint pbo;

GLFWwindow* initGLFW(int width, int height) {
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    exit(EXIT_FAILURE);
  }

  GLFWwindow* window = glfwCreateWindow(width, height, "CUDA GLFW Example", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    std::cerr << "Failed to create GLFW window" << std::endl;
    exit(EXIT_FAILURE);
  }

  glfwMakeContextCurrent(window);
  gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

  return window;
}

// CUDAとOpenGLのインタープレイ初期化
void initCUDAandGLInterop(int width, int height) {
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(char4) * width * height, nullptr, GL_DYNAMIC_DRAW);

  // OpenGLのバッファをCudaと共有する設定
  CUDA_CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&deviceResource, pbo, cudaGraphicsMapFlagsNone));
}

int main() {
  const int width = 1024;
  const int height = 1024;

  // GLFWウィンドウ初期化
  GLFWwindow* window = initGLFW(width, height);

  // CUDAとOpenGLのインタープレイ初期化
  initCUDAandGLInterop(width, height);

  // グリッドとブロックの設定
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  // メインループ
  while (!glfwWindowShouldClose(window)) {
    glClear(GL_COLOR_BUFFER_BIT);

    uchar4* cudaBitMap;
    size_t cudaSize;

    // CUDAとOpenGLのデータの同期
    CUDA_CHECK_ERROR(cudaGraphicsMapResources(1, &deviceResource, nullptr));
    CUDA_CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&cudaBitMap, &cudaSize, deviceResource));

    // CUDAカーネルの呼び出し
    launchCUDAKernel(cudaBitMap, width, height, (float)glfwGetTime());

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    CUDA_CHECK_ERROR(cudaGraphicsUnmapResources(1, &deviceResource, nullptr));

    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // 後片付け
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glDeleteBuffers(1, &pbo);
  CUDA_CHECK_ERROR(cudaGraphicsUnregisterResource(deviceResource));
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}