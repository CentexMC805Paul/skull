#!/usr/bin/env python3
# ============================================================
#  SKULL PYTHON PACKAGE SETUP v1.0.0
#  Build script for Python bindings
# ============================================================

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import platform
import subprocess

# ============================================================
#  PLATFORM DETECTION
# ============================================================

class PlatformInfo:
    """Detect platform and compiler settings"""
    
    @staticmethod
    def is_windows():
        return platform.system() == "Windows"
    
    @staticmethod
    def is_linux():
        return platform.system() == "Linux"
    
    @staticmethod
    def is_macos():
        return platform.system() == "Darwin"
    
    @staticmethod
    def get_compiler():
        if PlatformInfo.is_windows():
            return "msvc"
        else:
            return "unix"
    
    @staticmethod
    def get_extra_compile_args():
        args = []
        
        if PlatformInfo.is_windows():
            # MSVC specific flags
            args.extend([
                "/std:c++17",
                "/O2",
                "/W3",
                "/EHsc",
                "/arch:AVX2",
                "/DSKULL_USE_OPENCL",
            ])
        else:
            # GCC/Clang specific flags
            args.extend([
                "-std=c++17",
                "-O3",
                "-Wall",
                "-Wextra",
                "-mavx2",
                "-mfma",
                "-mbmi",
                "-mbmi2",
                "-DSKULL_USE_OPENCL",
            ])
        
        return args
    
    @staticmethod
    def get_extra_link_args():
        args = []
        
        if PlatformInfo.is_windows():
            args.extend([
                "OpenCL.lib",
            ])
        elif PlatformInfo.is_linux():
            args.extend([
                "-lOpenCL",
                "-lpthread",
            ])
        elif PlatformInfo.is_macos():
            args.extend([
                "-framework", "OpenCL",
            ])
        
        return args
    
    @staticmethod
    def get_include_dirs():
        dirs = [
            "src",
            os.path.join(os.path.dirname(__file__), "src"),
        ]
        
        # Try to find OpenCL headers
        opencl_paths = [
            "/usr/local/include",
            "/usr/include",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/include",
            "C:/Program Files (x86)/IntelSWTools/compiler/lib/intel64",
        ]
        
        for path in opencl_paths:
            if os.path.exists(os.path.join(path, "CL", "cl.h")):
                dirs.append(path)
            elif os.path.exists(os.path.join(path, "opencl.h")):
                dirs.append(path)
        
        return dirs
    
    @staticmethod
    def get_library_dirs():
        dirs = []
        
        # Try to find OpenCL libraries
        opencl_lib_paths = [
            "/usr/local/lib",
            "/usr/lib",
            "/usr/lib/x86_64-linux-gnu",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/lib/x64",
        ]
        
        for path in opencl_lib_paths:
            if os.path.exists(path):
                dirs.append(path)
        
        return dirs

# ============================================================
#  CUSTOM BUILD COMMAND
# ============================================================

class BuildWithCMake(build_ext):
    """Custom build command that uses CMake"""
    
    def build_extension(self, ext):
        # Get extension directory
        ext_dir = os.path.dirname(self.get_extension_fullpath(ext.name))
        
        # Create build directory
        build_dir = os.path.join(self.build_temp, "cmake_build")
        os.makedirs(build_dir, exist_ok=True)
        
        # Run CMake
        cmake_cmd = ["cmake"]
        if PlatformInfo.is_windows():
            cmake_cmd.extend([
                "-G", "Visual Studio 17 2022",
                "-A", "x64",
            ])
        else:
            cmake_cmd.extend([
                "-G", "Unix Makefiles",
            ])
        
        cmake_cmd.extend([
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSKULL_USE_OPENCL=ON",
            "-DSKULL_BUILD_PYTHON=ON",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPYTHON_INCLUDE_DIR={self.include_dirs[0]}",
        ])
        
        # Add source directory
        cmake_cmd.append(os.path.join(os.path.dirname(__file__), "src"))
        
        # Run CMake configuration
        subprocess.check_call(cmake_cmd, cwd=build_dir)
        
        # Build
        build_cmd = ["cmake", "--build", build_dir, "--config", "Release"]
        subprocess.check_call(build_cmd, cwd=build_dir)
        
        # Copy built library to extension directory
        if PlatformInfo.is_windows():
            lib_name = "skull_python.dll"
        elif PlatformInfo.is_linux():
            lib_name = "libskull_python.so"
        else:  # macOS
            lib_name = "libskull_python.dylib"
        
        lib_path = os.path.join(build_dir, "Release", lib_name)
        if os.path.exists(lib_path):
            import shutil
            shutil.copy(lib_path, os.path.join(ext_dir, ext.name + self.get_ext_suffix()))

# ============================================================
#  EXTENSION DEFINITION
# ============================================================

def get_extensions():
    """Get extension modules"""
    
    # Main Skull extension
    skull_ext = Extension(
        "skull._skull",
        sources=[
            "src/skull_python.cpp",
            "src/main.cpp",
            "src/lexer.h",
            "src/ast.h",
            "src/parser.h",
            "src/interpreter.h",
            "src/tensor.h",
            "src/tensor_info.h",
            "src/tokenizer.h",
            "src/trainer.h",
            "src/generator.h",
            "src/gpu.h",
            "src/config.h",
            "src/layers.h",
            "src/optimizer.h",
            "src/parallel.h",
            "src/model.h",
        ],
        include_dirs=PlatformInfo.get_include_dirs(),
        library_dirs=PlatformInfo.get_library_dirs(),
        libraries=PlatformInfo.get_extra_link_args(),
        extra_compile_args=PlatformInfo.get_extra_compile_args(),
        extra_link_args=PlatformInfo.get_extra_link_args(),
        language="c++",
    )
    
    return [skull_ext]

# ============================================================
#  CUDA SUPPORT
# ============================================================

def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        return tf.config.list_physical_devices('GPU')
    except ImportError:
        pass
    
    # Check for nvcc
    try:
        subprocess.check_call(["nvcc", "--version"], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_cuda_extensions():
    """Get CUDA extensions if available"""
    if not check_cuda():
        print("CUDA not found, skipping CUDA extensions")
        return []
    
    print("CUDA found, building CUDA extensions")
    
    # Find CUDA include and library paths
    cuda_include = None
    cuda_lib = None
    
    # Try common CUDA paths
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12.0",
        "/usr/local/cuda-11.8",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8",
    ]
    
    for path in cuda_paths:
        if os.path.exists(os.path.join(path, "include", "cuda.h")):
            cuda_include = os.path.join(path, "include")
        if os.path.exists(os.path.join(path, "lib64", "libcudart.so")):
            cuda_lib = os.path.join(path, "lib64")
        if os.path.exists(os.path.join(path, "lib", "x64", "cudart64_100.dll")):
            cuda_lib = os.path.join(path, "lib", "x64")
    
    if not cuda_include or not cuda_lib:
        print("CUDA paths not found, skipping CUDA extensions")
        return []
    
    # CUDA extension
    cuda_ext = Extension(
        "skull._cuda",
        sources=["src/gpu_cuda.cu"],
        include_dirs=[cuda_include, "src"],
        library_dirs=[cuda_lib],
        libraries=["cudart", "cuda"],
        extra_compile_args=[
            "-std=c++17",
            "-O3",
            "-DSKULL_USE_CUDA",
        ],
        extra_link_args=[],
        language="c++",
    )
    
    return [cuda_ext]

# ============================================================
#  SETUP CONFIGURATION
# ============================================================

# Read long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Get version
version = "1.0.0"

# Get all extensions
extensions = get_extensions()
cuda_extensions = get_cuda_extensions()
all_extensions = extensions + cuda_extensions

# Setup configuration
setup(
    name="skull",
    version=version,
    description="Skull - A programming language for training language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Skull Contributors",
    author_email="",
    url="https://github.com/CentexMC805Paul/skull",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Compilers",
    ],
    python_requires=">=3.8",
    
    # Packages
    packages=find_packages(),
    
    # Extensions
    ext_modules=all_extensions,
    
    # Package data
    package_data={
        "skull": ["*.skull", "*.txt", "*.md", "*.json", "*.jsonl"],
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "skull=skull.cli:main",
        ],
    },
    
    # Install requires
    install_requires=[
        "numpy>=1.20.0",
        "pybind11>=2.10.0",
    ],
    
    # Extras
    extras_require={
        "cuda": ["torch>=2.0.0"],
        "opencl": ["pyopencl>=2023.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
    
    # CMake build
    cmdclass={
        "build_ext": BuildWithCMake,
    },
)
