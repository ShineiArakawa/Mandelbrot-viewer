cmake_minimum_required(VERSION 3.24)
project(Mandelbrot-viewer)

# #################################################
# #### Compile Options ############################
# #################################################
set(CMAKE_CXX_STANDARD 17)

if(UNIX)
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
elseif(WIN32)
  set(CMAKE_CXX_FLAGS_DEBUG "-Wall")
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 -DNDEBUG")
endif()

# #################################################
# #### Envirionment Variables #####################
# #################################################
# #### Set environment variables in project
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(PROJECT_EXTERNAL_DIR ${PROJECT_SOURCE_DIR}/external)

# #################################################
# #### External Libraries #########################
# #################################################
# #### OpenGL
find_package(OpenGL REQUIRED)

# #### OpenMP
find_package(OpenMP REQUIRED)

if(OpenMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# #### GLAD
set(GLAD_INCLUDE_DIR ${PROJECT_EXTERNAL_DIR}/glad/include)

# #### GLFW
set(GLFW_INCLUDE_DIR ${PROJECT_EXTERNAL_DIR}/glfw/include)

# #### GLM
set(GLM_INCLUDE_DIR ${PROJECT_EXTERNAL_DIR}/glm)

# #### X11
if(UNIX)
  find_package(X11 REQUIRED)
endif(UNIX)

# #### Dear Imgui
set(IMGUI_INCLUDE_DIR ${PROJECT_EXTERNAL_DIR}/imgui)
set(IMGUI_SOURCE_DIR ${PROJECT_EXTERNAL_DIR}/imgui)
set(IMGUI_BACKENDS_DIR ${IMGUI_SOURCE_DIR}/backends)

set(
  IMGUI_SOURCE_FILES
  ${IMGUI_SOURCE_DIR}/imgui.cpp
  ${IMGUI_SOURCE_DIR}/imgui_demo.cpp
  ${IMGUI_SOURCE_DIR}/imgui_draw.cpp
  ${IMGUI_SOURCE_DIR}/imgui_tables.cpp
  ${IMGUI_SOURCE_DIR}/imgui_widgets.cpp
  ${IMGUI_SOURCE_DIR}/imgui_demo.cpp
  ${IMGUI_BACKENDS_DIR}/imgui_impl_glfw.cpp
  ${IMGUI_BACKENDS_DIR}/imgui_impl_opengl3.cpp
)

# #### stb
set(STB_INCLUDE_DIR ${PROJECT_EXTERNAL_DIR}/stb)

# #### stb
set(ARGPARSE_INCLUDE_DIR ${PROJECT_EXTERNAL_DIR}/argparse/include)

# #### Add subdirectory
add_subdirectory("src")
add_subdirectory("external")

# #################################################
# #### Message ####################################
# #################################################
message("# =======================================================================================================")
message("#  '${PROJECT_NAME}' build configuration ")
message("# =======================================================================================================")
message("#  [CMake]")
message("#    CMake version            : ${CMAKE_VERSION}")
message("#    CMake build type         : ${CMAKE_BUILD_TYPE}")
message("# ")
message("#  [C/C++]")
message("#    C++ Compiler             : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} ${CMAKE_CXX_COMPILER_WRAPPER} ${CMAKE_CXX_COMPILER}")
message("#    CXX STANDARD             : ${CMAKE_CXX_STANDARD}")
message("# ")
message("#  [OpenMP]")
message("#    OpenMP_C_FLAGS           : ${OpenMP_C_FLAGS}")
message("#    OpenMP_CXX_FLAGS         : ${OpenMP_CXX_FLAGS}")
message("# ")
message("#  [Environment Variables]")
message("#    PROJECT_INCLUDE_DIR      : ${PROJECT_INCLUDE_DIR}")
message("#    PROJECT_EXTERNAL_DIR     : ${PROJECT_EXTERNAL_DIR}")
message("# ")
message("#    GLAD_INCLUDE_DIR         : ${GLAD_INCLUDE_DIR}")
message("#    GLFW_INCLUDE_DIR         : ${GLFW_INCLUDE_DIR}")
message("#    GLM_INCLUDE_DIR          : ${GLM_INCLUDE_DIR}")
message("#    IMGUI_INCLUDE_DIR        : ${IMGUI_INCLUDE_DIR}")
message("#    IMGUI_SOURCE_DIR         : ${IMGUI_SOURCE_DIR}")
message("#    IMGUI_BACKENDS_DIR       : ${IMGUI_BACKENDS_DIR}")
message("#    STB_INCLUDE_DIR          : ${STB_INCLUDE_DIR}")
message("#    ARGPARSE_INCLUDE_DIR     : ${ARGPARSE_INCLUDE_DIR}")
message("# =======================================================================================================")
