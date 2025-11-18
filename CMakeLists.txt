# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(MatmulViz VERSION 1.0 LANGUAGES CXX CUDA)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define a common output directory
set(COMMON_OUTPUT_DIR "${CMAKE_BINARY_DIR}/output")

# Set output directories for different types of binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${COMMON_OUTPUT_DIR}/bin")  # For executables
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${COMMON_OUTPUT_DIR}/lib")  # For shared libraries
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${COMMON_OUTPUT_DIR}/lib")  # For static libraries

add_subdirectory(SFML)

# Include header files
include_directories(${PROJECT_SOURCE_DIR}/include)

link_directories(${PROJECT_SOURCE_DIR}/lib)

set(SOURCES ${PROJECT_SOURCE_DIR}/GameOfLife.cpp ${PROJECT_SOURCE_DIR}/matrix.cu)

add_executable(MatmulViz ${SOURCES})

include_directories(${PROJECT_SOURCE_DIR}/SFML/include)

link_directories(${PROJECT_SOURCE_DIR}/SFML/lib)

target_link_libraries(MatmulViz PUBLIC sfml-graphics sfml-system sfml-window)
