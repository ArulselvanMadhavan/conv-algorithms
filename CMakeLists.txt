cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(VERBOSE ON)
set(LLVM_ENABLE_LLD ON)

project(conv_algorithms
    VERSION 0.0.1
    DESCRIPTION "Try different conv algorithms"
    HOMEPAGE_URL "https://github.com/ArulselvanMadhavan/conv-algorithms"
)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# LLVM
find_package(LLVM REQUIRED HINTS "${LLVM_CMAKE_PATH}")
list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})
include(AddLLVM)
include(HandleLLVMOptions)
include_directories("${LLVM_BINARY_DIR}/include" "${LLVM_INCLUDE_DIR}")
link_directories("${LLVM_LIBRARY_DIR}")

# Libtorch
set(PYTORCH_VERSION "1.13.0")
find_package(Torch ${PYTORCH_VERSION} EXACT QUIET PATHS
"${CMAKE_SOURCE_DIR}/libtorch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(NOT Torch_FOUND)
  unset(Torch_FOUND)
  include(fetch_libtorch)
endif()

# Target kn2row
set(EXECUTABLE_NAME kn2row)
add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE src/kn2row.cpp)
target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES})
set_target_properties(${EXECUTABLE_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES)

    



