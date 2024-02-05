project(MandelbrotGUI LANGUAGES CXX CUDA)

# ############################# Executable C++ ##############################
add_executable(
    ${PROJECT_NAME}
    "main.cpp"
    ${IMGUI_SOURCE_FILES}
    $<TARGET_OBJECTS:Common>
    $<TARGET_OBJECTS:MandelbrotModel>
    $<TARGET_OBJECTS:MandelbrotModelKernel>
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
    ${ARGPARSE_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
    glad
    glfw
    OpenGL::GL
    OpenGL::GLU
    stdc++fs
)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_86,code=sm_86>)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION
    ON
)

add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_SOURCE_DIR}/bin/
)