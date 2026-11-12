cmake_minimum_required(VERSION 3.16)
project(tcmma_model CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

add_library(tcmma STATIC
  src/formats.cpp
  src/fda.cpp
  src/gen.cpp
  src/probe_io.cpp)
target_include_directories(tcmma PUBLIC include)
set_target_properties(tcmma PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(tcmma_cli tools/tcmma_cli.cpp)
target_link_libraries(tcmma_cli PRIVATE tcmma)
set_target_properties(tcmma_cli PROPERTIES OUTPUT_NAME tcmma)

add_executable(test_model tests/test_model.cpp)
target_link_libraries(test_model PRIVATE tcmma)

enable_testing()
add_test(NAME model_unit_tests COMMAND test_model)
