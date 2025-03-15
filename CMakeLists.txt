cmake_minimum_required(VERSION 3.29.3)
project(TENSOR CXX)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(extern)
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(apps)
