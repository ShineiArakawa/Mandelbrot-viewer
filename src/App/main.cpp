#include <App/main.hpp>
#include <Common/FileUtil.hpp>
#include <Common/GUI.hpp>
#include <Mandelbrot/model.hpp>

static bool isDragging = false;
ImVec2 oldPos;
ImVec2 newPos;
ImVec2 oldRangeX;
ImVec2 oldRangeY;
ImVec2 newRangeX;
ImVec2 newRangeY;

int main(int argc, char** argv) {
  // Parse args
  argparse::ArgumentParser program("Mandelbrot-viewer");

  program.add_argument("--offline").default_value(false).implicit_value(true);
  program.add_argument("--minX").default_value(-2.0).scan<'g', double>();
  program.add_argument("--maxX").default_value(2.0).scan<'g', double>();
  program.add_argument("--minY").default_value(-2.0).scan<'g', double>();
  program.add_argument("--maxY").default_value(2.0).scan<'g', double>();
  program.add_argument("--nFrames").default_value(1000).scan<'i', int>();
  program.add_argument("--delta").default_value(0.1).scan<'g', double>();
  program.add_argument("--saveDir").default_value("");

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  int returnState = 1;
  if (program["--offline"] == true) {
    returnState = offlineRender(program);
  } else {
    returnState = launchWindow();
  }

  return 0;
}

int offlineRender(argparse::ArgumentParser& parser) {
  std::cout << "Start offline rendering" << std::endl;

  auto mandelbrotModel = std::make_shared<mandel::model::MandelbrotModel>(true);

  mandelbrotModel->minX = parser.get<double>("minX");
  mandelbrotModel->maxX = parser.get<double>("maxX");
  mandelbrotModel->minY = parser.get<double>("minY");
  mandelbrotModel->maxY = parser.get<double>("maxY");
  const int nFrames = parser.get<int>("nFrames");
  const double delta = parser.get<double>("delta");
  const std::string saveDir = mandel::fs::FileUtil::absPath(parser.get<std::string>("saveDir"));

  for (int iFrame = 0; iFrame < nFrames; iFrame++) {
    double bandwidthX = mandelbrotModel->maxX - mandelbrotModel->minX;
    double bandwidthY = mandelbrotModel->maxY - mandelbrotModel->minY;
    mandelbrotModel->minX = mandelbrotModel->minX + bandwidthX * delta;
    mandelbrotModel->maxX = mandelbrotModel->maxX - bandwidthX * delta;
    mandelbrotModel->minY = mandelbrotModel->minY + bandwidthY * delta;
    mandelbrotModel->maxY = mandelbrotModel->maxY - bandwidthY * delta;

    mandelbrotModel->update();

    std::string filePath = mandel::fs::FileUtil::join(saveDir, std::to_string(iFrame) + ".png");
    mandel::GUI::saveImage(mandelbrotModel->TEXTURE_BUFFER_WIDTH, mandelbrotModel->TEXTURE_BUFFER_HEIGHT, 4, mandelbrotModel->bytePixelsBuffer, filePath);
  }

  return 0;
}

int launchWindow() {
  if (glfwInit() == GLFW_FALSE) {
    fprintf(stderr, "Initialization failed!\n");
    return 1;
  }

  // Create a window
  GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, NULL, NULL);
  if (window == NULL) {
    glfwTerminate();
    fprintf(stderr, "Window creation failed!\n");
    return 1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);  // Enable vsync

  // OpenGL 3.x/4.xの関数をロードする (glfwMakeContextCurrentの後でないといけない)
  const int version = gladLoadGL();
  if (version == 0) {
    fprintf(stderr, "Failed to load OpenGL 3.x/4.x libraries!\n");
    return 1;
  }

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = NULL;
  io.LogFilename = NULL;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
  // io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;      // IF using Docking Branch

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window, true);  // Second param install_callback=true will install GLFW callbacks and chain to existing ones.
  ImGui_ImplOpenGL3_Init();

  // Enable depth testing
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_TEXTURE_2D);
  auto mandelbrotModel = std::make_shared<mandel::model::MandelbrotModel>();

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    glClear(GL_COLOR_BUFFER_BIT);

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    {
      // Control window
      if (ImGui::BeginViewportSideBar("Side bar", ImGui::GetWindowViewport(), ImGuiDir_::ImGuiDir_Left, SIDEBAR_WIDTH, true)) {
        ImGui::BeginChild("Setting", ImVec2(0, SETTING_TAB_HEIGHT));
        if (ImGui::BeginTabBar("Setting")) {
          if (ImGui::BeginTabItem("Range")) {
            ImGui::InputDouble("Min X", &mandelbrotModel->minY);
            ImGui::InputDouble("Max X", &mandelbrotModel->maxY);
            ImGui::InputDouble("Min Y", &mandelbrotModel->minX);
            ImGui::InputDouble("Max Y", &mandelbrotModel->maxX);
            ImGui::EndTabItem();
          }

          if (ImGui::BeginTabItem("Calculation")) {
            ImGui::InputInt("Max Iter", &mandelbrotModel->maxIter);
            ImGui::InputDouble("Threshold", &mandelbrotModel->threshold);
            ImGui::InputDouble("Alpha Coeff", &mandelbrotModel->alphaCoeff);
            ImGui::EndTabItem();
          }

          if (ImGui::BeginTabItem("Render")) {
            ImGui::Checkbox("Smoothing", &mandelbrotModel->isEnabledSmoothing);
            ImGui::Checkbox("Sinusoidal Color", &mandelbrotModel->isEnabledSinuidalColor);
            ImGui::InputDouble("Density", &mandelbrotModel->density);
            ImGui::Checkbox("Super Sample", &mandelbrotModel->isEnabledSuperSampling);
            ImGui::InputInt("Super Sample Factor", &mandelbrotModel->superSampleFactor);
            ImGui::EndTabItem();
          }

          ImGui::EndTabBar();
        }
        ImGui::EndChild();

        ImGui::BeginChild("Setting2");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        ImGui::EndChild();

        ImGui::End();
      }
    }

    ImVec2 validImageSize;
    ImVec2 validImageOrigin;
    ImVec2 validImageCoordMin;
    ImVec2 validImageCoordMax;

    {
      // Image window
      ImVec2 viewportPos = ImGui::GetWindowViewport()->Pos;
      ImGui::SetNextWindowPos(ImVec2(viewportPos.x + SIDEBAR_WIDTH, viewportPos.y));
      ImVec2 windowSize = ImVec2(ImGui::GetWindowViewport()->Size.x - SIDEBAR_WIDTH, ImGui::GetWindowViewport()->Size.y);
      ImGui::SetNextWindowSizeConstraints(windowSize, windowSize);

      ImGui::Begin("View");
      {
        const float window_width = ImGui::GetContentRegionAvail().x;
        const float window_height = ImGui::GetContentRegionAvail().y;
        const float image_width = mandelbrotModel->getTexture()->getWidth();
        const float image_height = mandelbrotModel->getTexture()->getHeight();

        if (window_width < window_height) {
          validImageSize.x = window_width;
          validImageSize.y = image_height / image_width * window_width;
          validImageOrigin.x = 0.0;
          validImageOrigin.y = (window_height - validImageSize.y) / 2.0;
        } else {
          validImageSize.x = image_width / image_height * window_height;
          validImageSize.y = window_height;
          validImageOrigin.x = (window_width - validImageSize.x) / 2.0;
          validImageOrigin.y = 0.0;
        }

        ImVec2 pos = ImGui::GetCursorScreenPos();
        validImageCoordMin.x = pos.x + validImageOrigin.x;
        validImageCoordMin.y = pos.y + validImageOrigin.y;
        validImageCoordMax.x = pos.x + validImageOrigin.x + validImageSize.x;
        validImageCoordMax.y = pos.y + validImageOrigin.y + validImageSize.y;

        ImGui::GetWindowDrawList()->AddImage(
            (void*)(intptr_t)mandelbrotModel->getTextureID(),
            validImageCoordMin,
            validImageCoordMax,
            ImVec2(0, 0),
            ImVec2(1, 1));
      }
      ImGui::End();
    }

    {
      ImVec2 mousePos = ImGui::GetMousePos();

      if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && validImageCoordMin.x < mousePos.x && mousePos.x < validImageCoordMax.x && validImageCoordMin.y < mousePos.y && mousePos.y < validImageCoordMax.y) {
        if (!isDragging) {
          isDragging = true;
          oldPos = mousePos;
          newPos = mousePos;

          oldRangeX = ImVec2(mandelbrotModel->minX, mandelbrotModel->maxX);
          oldRangeY = ImVec2(mandelbrotModel->minY, mandelbrotModel->maxY);
          newRangeX = ImVec2(mandelbrotModel->minX, mandelbrotModel->maxX);
          newRangeY = ImVec2(mandelbrotModel->minY, mandelbrotModel->maxY);
        }
      } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        if (isDragging) {
          isDragging = false;
        }
      }

      if (isDragging) {
        newPos = mousePos;

        const double dx = newPos.x - oldPos.x;
        const double dy = newPos.y - oldPos.y;
        double widthPerPixel = (mandelbrotModel->maxX - mandelbrotModel->minX) / mandelbrotModel->getTexture()->getWidth();
        double heightPerPixel = (mandelbrotModel->maxY - mandelbrotModel->minY) / mandelbrotModel->getTexture()->getHeight();

        mandelbrotModel->minX = oldRangeX.x - dy * widthPerPixel;
        mandelbrotModel->maxX = oldRangeX.y - dy * widthPerPixel;
        mandelbrotModel->minY = oldRangeY.x - dx * heightPerPixel;
        mandelbrotModel->maxY = oldRangeY.y - dx * heightPerPixel;
      }

      if ((MOUSE_WHEEL_THRESHOLD * MOUSE_WHEEL_THRESHOLD < io.MouseWheel * io.MouseWheel) && validImageCoordMin.x < mousePos.x && mousePos.x < validImageCoordMax.x && validImageCoordMin.y < mousePos.y && mousePos.y < validImageCoordMax.y) {
        double bandwidthX = mandelbrotModel->maxX - mandelbrotModel->minX;
        double bandwidthY = mandelbrotModel->maxY - mandelbrotModel->minY;
        double factor = io.MouseWheel > 0.0 ? 1.0 : -1.0;
        mandelbrotModel->minX = mandelbrotModel->minX + bandwidthX * MOUSE_WHEEL_DELTA * factor;
        mandelbrotModel->maxX = mandelbrotModel->maxX - bandwidthX * MOUSE_WHEEL_DELTA * factor;
        mandelbrotModel->minY = mandelbrotModel->minY + bandwidthY * MOUSE_WHEEL_DELTA * factor;
        mandelbrotModel->maxY = mandelbrotModel->maxY - bandwidthY * MOUSE_WHEEL_DELTA * factor;
      }
    }

    // Rendering
    mandelbrotModel->update();

    ImGui::Render();

    // mandelbrotModel->update();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);

    glfwSwapBuffers(window);
  }

  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}