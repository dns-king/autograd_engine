cmake_minimum_required(VERSION 3.20)
project(autograd_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(autograd src/value.cpp src/mlp.cpp)
target_include_directories(autograd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(MSVC)
  target_compile_options(autograd PRIVATE /W4 /permissive-)
else()
  target_compile_options(autograd PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(autograd_demo src/main.cpp)
target_link_libraries(autograd_demo PRIVATE autograd)

enable_testing()
add_executable(autograd_tests tests/test_autograd.cpp)
target_link_libraries(autograd_tests PRIVATE autograd)
add_test(NAME autograd_tests COMMAND autograd_tests)
