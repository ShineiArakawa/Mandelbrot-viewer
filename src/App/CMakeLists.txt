project(Mandelbrot LANGUAGES CXX)

# ############################# Executable C++ ##############################
add_executable(
    ${PROJECT_NAME}
    "main.cpp"
    ${IMGUI_SOURCE_FILES}
    $<TARGET_OBJECTS:Common>
    $<TARGET_OBJECTS:MandelbrotModel>
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
)

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
    glad
    glfw
    OpenGL::GL
    OpenGL::GLU
)

if(UNIX)
    target_link_libraries(
        ${EXE_PROJECT_NAME}
        X11
    )
endif(UNIX)

add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_SOURCE_DIR}/bin/
)