project(MandelbrotModel LANGUAGES CXX)

# ############################# Executable C++ ##############################
add_library(
    ${PROJECT_NAME}
    STATIC
    "model.cpp"
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${PROJECT_INCLUDE_DIR}
    ${GLAD_INCLUDE_DIR}
    ${GLFW_INCLUDE_DIR}
    ${GLM_INCLUDE_DIR}
    ${IMGUI_INCLUDE_DIR}
    ${STB_INCLUDE_DIR}
)
