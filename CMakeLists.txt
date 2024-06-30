cmake_minimum_required(VERSION 3.18)
project(ToyML)

find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/native)