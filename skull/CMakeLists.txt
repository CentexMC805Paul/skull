cmake_minimum_required(VERSION 3.15)
project(Skull LANGUAGES CXX)

# ============================================================
#  SKULL CMAKE BUILD SYSTEM v1.0.0
#  Cross-Platform: Windows, Linux, macOS
#  Supports: CPU (AVX2/SSE), OpenCL (GPU), CUDA (NVIDIA)
# ============================================================

# Version
set(SKULL_VERSION "1.0.0")
message(STATUS "Skull v${SKULL_VERSION} - Cross-Platform Build")

# C++ Standard (C++17 required)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
#  OPTIMIZATION FLAGS
# ============================================================

# Release Build optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -DNDEBUG)
    message(STATUS "Build Type: Release (Optimized)")
else()
    add_compile_options(-O2 -g)
    message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
endif()

# Architecture-specific optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
    # AVX2 support (modern x86_64)
    if(MSVC)
        add_compile_options(/arch:AVX2)
    else()
        add_compile_options(-mavx2 -mfma -mbmi -mbmi2)
    endif()
    message(STATUS "AVX2 optimizations enabled")
    add_definitions(-DSKULL_AVX2=1)
endif()

# ============================================================
#  GPU BACKEND OPTIONS
# ============================================================

# OpenCL Support (Cross-Platform GPU: AMD, NVIDIA, Intel)
option(SKULL_USE_OPENCL "Enable OpenCL GPU support" ON)
if(SKULL_USE_OPENCL)
    find_package(OpenCL REQUIRED)
    if(OpenCL_FOUND)
        message(STATUS "OpenCL found: ${OpenCL_INCLUDE_DIRS}")
        add_definitions(-DSKULL_USE_OPENCL=1 -DSKULL_GPU_AVAILABLE=1)
        include_directories(${OpenCL_INCLUDE_DIRS})
        link_directories(${OpenCL_LIBRARY_DIRS})
    else()
        message(WARNING "OpenCL not found! Falling back to CPU only")
        add_definitions(-DSKULL_USE_OPENCL=0 -DSKULL_GPU_AVAILABLE=0)
    endif()
endif()

# CUDA Support (NVIDIA only)
option(SKULL_USE_CUDA "Enable CUDA GPU support (NVIDIA)" OFF)
if(SKULL_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
        add_definitions(-DSKULL_USE_CUDA=1)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        include_directories(${CUDAToolkit_INCLUDE_DIRS})
    else()
        message(WARNING "CUDA not found! Falling back to CPU/OpenCL")
        add_definitions(-DSKULL_USE_CUDA=0)
    endif()
endif()

# ============================================================
#  PLATFORM-SPECIFIC SETTINGS
# ============================================================

# Windows
if(WIN32)
    message(STATUS "Platform: Windows")
    if(MSVC)
        add_compile_options(/W3 /EHsc /MP)
        # Disable warnings for OpenCL/CUDA
        add_compile_options(/wd4244 /wd4305 /wd4996)
    endif()
    # Windows-specific libraries
    set(PLATFORM_LIBS OpenCL)

# Linux
elseif(UNIX AND NOT APPLE)
    message(STATUS "Platform: Linux")
    add_compile_options(-Wall -Wextra -Wpedantic -pthread)
    # Linux-specific libraries
    find_package(Threads REQUIRED)
    set(PLATFORM_LIBS Threads::Threads OpenCL)

# macOS
elseif(APPLE)
    message(STATUS "Platform: macOS")
    add_compile_options(-Wall -Wextra -Wpedantic -pthread)
    # macOS OpenCL framework
    find_library(OPENCL_LIBRARY OpenCL)
    find_path(OPENCL_INCLUDE_DIR OpenCL/opencl.h)
    if(OPENCL_LIBRARY AND OPENCL_INCLUDE_DIR)
        include_directories(${OPENCL_INCLUDE_DIR})
        set(PLATFORM_LIBS ${OPENCL_LIBRARY})
    else()
        message(WARNING "OpenCL not found on macOS")
    endif()
    find_library(COCOA_LIBRARY Cocoa)
    find_library(IOKIT_LIBRARY IOKit)
    find_library(CARBON_LIBRARY Carbon)
    list(APPEND PLATFORM_LIBS ${COCOA_LIBRARY} ${IOKIT_LIBRARY} ${CARBON_LIBRARY})
endif()

# ============================================================
#  SOURCES
# ============================================================

# Main source files
set(SKULL_SOURCES
    src/main.cpp
    src/config.h
    src/lexer.h
    src/ast.h
    src/parser.h
    src/interpreter.h
    src/tensor.h
    src/tensor_info.h
    src/tokenizer.h
    src/trainer.h
    src/generator.h
    src/gpu.h
    src/layers.h
    src/optimizer.h
    src/parallel.h
    src/model.h
)

# CUDA sources (if enabled)
if(SKULL_USE_CUDA AND CUDAToolkit_FOUND)
    list(APPEND SKULL_SOURCES src/gpu_cuda.cu)
    message(STATUS "CUDA sources added")
endif()

# Python bindings (if enabled)
option(SKULL_BUILD_PYTHON "Build Python bindings" OFF)
if(SKULL_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    if(Python3_FOUND)
        message(STATUS "Python3 found: ${Python3_VERSION}")
        list(APPEND SKULL_SOURCES src/skull_python.cpp)
        include_directories(${Python3_INCLUDE_DIRS})
        add_definitions(-DSKULL_BUILD_PYTHON=1)
    else()
        message(WARNING "Python3 not found! Python bindings will not be built")
    endif()
endif()

# CUDA sources (if enabled)
if(SKULL_USE_CUDA AND CUDAToolkit_FOUND)
    list(APPEND SKULL_SOURCES src/gpu_cuda.cu)
    message(STATUS "CUDA sources added")
endif()

# ============================================================
#  EXECUTABLE
# ============================================================

add_executable(skull ${SKULL_SOURCES})

# Include directories
target_include_directories(skull PRIVATE src)

# Link libraries
target_link_libraries(skull PRIVATE ${PLATFORM_LIBS})

if(SKULL_BUILD_PYTHON AND Python3_FOUND)
    target_link_libraries(skull PRIVATE ${Python3_LIBRARIES})
endif()

if(SKULL_USE_CUDA AND CUDAToolkit_FOUND)
    target_link_libraries(skull PRIVATE CUDA::cudart CUDA::cuda_driver)
    set_target_properties(skull PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(SKULL_USE_OPENCL AND OpenCL_FOUND)
    target_link_libraries(skull PRIVATE ${OpenCL_LIBRARIES})
endif()

if(SKULL_USE_CUDA AND CUDAToolkit_FOUND)
    target_link_libraries(skull PRIVATE CUDA::cudart CUDA::cuda_driver)
    set_target_properties(skull PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# ============================================================
#  INSTALLATION
# ============================================================

# Install executable
install(TARGETS skull
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install examples
install(DIRECTORY examples
    DESTINATION share/skull
)

# ============================================================
#  CUSTOM TARGETS
# ============================================================

# Test target
add_custom_target(test
    COMMAND skull examples/hello.skull
    COMMAND skull examples/tensor_test.skull
    COMMAND skull examples/train_demo.skull
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS skull
    COMMENT "Running Skull test examples"
)

# ============================================================
#  OUTPUT MESSAGES
# ============================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Skull Configuration Summary:")
message(STATUS "  Version: ${SKULL_VERSION}")
message(STATUS "  C++ Standard: C++17")
message(STATUS "  AVX2: ${SKULL_AVX2}")
message(STATUS "  OpenCL: ${SKULL_USE_OPENCL}")
message(STATUS "  CUDA: ${SKULL_USE_CUDA}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "========================================")
message(STATUS "")
