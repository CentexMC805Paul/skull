#pragma once
// ============================================================
//  SKULL CONFIGURATION HEADER v1.0.0
//  Central configuration for all Skull components
// ============================================================

#include <string>
#include <vector>
#include <map>

// ============================================================
//  VERSION INFO
// ============================================================
#define SKULL_VERSION_MAJOR 1
#define SKULL_VERSION_MINOR 0
#define SKULL_VERSION_PATCH 0
#define SKULL_VERSION_STRING "1.0.0"

// ============================================================
//  PLATFORM DETECTION
// ============================================================
#if defined(_WIN32) || defined(_WIN64)
    #define SKULL_PLATFORM_WINDOWS 1
    #define SKULL_PLATFORM_LINUX 0
    #define SKULL_PLATFORM_MACOS 0
#elif defined(__APPLE__) || defined(__MACH__)
    #define SKULL_PLATFORM_WINDOWS 0
    #define SKULL_PLATFORM_LINUX 0
    #define SKULL_PLATFORM_MACOS 1
#else
    #define SKULL_PLATFORM_WINDOWS 0
    #define SKULL_PLATFORM_LINUX 1
    #define SKULL_PLATFORM_MACOS 0
#endif

// ============================================================
//  ARCHITECTURE DETECTION
// ============================================================
#if defined(__x86_64__) || defined(_M_X64)
    #define SKULL_ARCH_X86_64 1
    #define SKULL_ARCH_ARM 0
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define SKULL_ARCH_X86_64 0
    #define SKULL_ARCH_ARM 1
#else
    #define SKULL_ARCH_X86_64 0
    #define SKULL_ARCH_ARM 0
#endif

// ============================================================
//  COMPILER DETECTION
// ============================================================
#if defined(_MSC_VER)
    #define SKULL_COMPILER_MSVC 1
    #define SKULL_COMPILER_GCC 0
    #define SKULL_COMPILER_CLANG 0
#elif defined(__GNUC__)
    #define SKULL_COMPILER_MSVC 0
    #if defined(__clang__)
        #define SKULL_COMPILER_GCC 0
        #define SKULL_COMPILER_CLANG 1
    #else
        #define SKULL_COMPILER_GCC 1
        #define SKULL_COMPILER_CLANG 0
    #endif
#else
    #define SKULL_COMPILER_MSVC 0
    #define SKULL_COMPILER_GCC 0
    #define SKULL_COMPILER_CLANG 0
#endif

// ============================================================
//  FEATURE FLAGS
// ============================================================

// AVX2 Support
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
    #define SKULL_AVX2 1
#else
    #define SKULL_AVX2 0
#endif

// SSE4.2 Support
#if defined(__SSE4_2__) || (defined(_MSC_VER) && defined(__SSE4_2__))
    #define SKULL_SSE42 1
#else
    #define SKULL_SSE42 0
#endif

// OpenCL Support
#ifdef SKULL_USE_OPENCL
    #define SKULL_GPU_OPENCL 1
#else
    #define SKULL_GPU_OPENCL 0
#endif

// CUDA Support
#ifdef SKULL_USE_CUDA
    #define SKULL_GPU_CUDA 1
#else
    #define SKULL_GPU_CUDA 0
#endif

// Metal Support (macOS)
#ifdef SKULL_USE_METAL
    #define SKULL_GPU_METAL 1
#else
    #define SKULL_GPU_METAL 0
#endif

// Mixed Precision Support (FP16)
#ifdef SKULL_USE_FP16
    #define SKULL_FP16 1
#else
    #define SKULL_FP16 0
#endif

// ============================================================
//  PRECISION SETTINGS
// ============================================================

// Default precision for computations
enum class PrecisionMode {
    FP32,   // Single precision (default)
    FP16,   // Half precision (faster, less memory)
    BF16,   // BFloat16 (like FP16 but with FP32 exponent)
    FP64    // Double precision (slower, more memory)
};

// Global precision mode (can be changed at runtime)
extern PrecisionMode g_precision_mode;

// Set global precision mode
inline void set_precision_mode(PrecisionMode mode) {
    g_precision_mode = mode;
}

// Get current precision mode
inline PrecisionMode get_precision_mode() {
    return g_precision_mode;
}

// ============================================================
//  GPU BACKEND SELECTION
// ============================================================

enum class GPUBackend {
    CPU,        // CPU only (SIMD)
    OPENCL,    // OpenCL (AMD, NVIDIA, Intel)
    CUDA,       // CUDA (NVIDIA only)
    METAL,      // Metal (Apple)
    AUTO        // Auto-select best available
};

// Global GPU backend (can be changed at runtime)
extern GPUBackend g_gpu_backend;

// Set GPU backend
inline void set_gpu_backend(GPUBackend backend) {
    g_gpu_backend = backend;
}

// Get current GPU backend
inline GPUBackend get_gpu_backend() {
    return g_gpu_backend;
}

// ============================================================
//  MODEL ARCHITECTURE TYPES
// ============================================================

enum class ModelArchitecture {
    FEEDFORWARD,      // Simple feedforward network
    TRANSFORMER,      // Transformer with self-attention
    LSTM,             // Long Short-Term Memory
    GRU,             // Gated Recurrent Unit
    CONV1D,          // 1D Convolutional
    CUSTOM           // User-defined architecture
};

// ============================================================
//  ACTIVATION FUNCTIONS
// ============================================================

enum class ActivationFunction {
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    GELU,
    SWISH,
    SOFTMAX,
    NONE
};

// ============================================================
//  OPTIMIZER TYPES
// ============================================================

enum class OptimizerType {
    SGD,            // Stochastic Gradient Descent
    ADAM,           // Adaptive Moment Estimation
    ADAMW,          // Adam with Weight Decay
    RMSPROP,        // Root Mean Square Propagation
    ADAGRAD,        // Adaptive Gradient Algorithm
    ADADELTA,       // Adaptive Delta
    LION            // Lion optimizer (new, efficient)
};

// ============================================================
//  LOSS FUNCTIONS
// ============================================================

enum class LossFunction {
    MSE,            // Mean Squared Error
    CROSS_ENTROPY, // Cross-Entropy (for classification)
    BCE,            // Binary Cross-Entropy
    HINGE,          // Hinge Loss
    KL_DIVERGENCE,  // Kullback-Leibler Divergence
    COSINE          // Cosine Similarity
};

// ============================================================
//  TOKENIZER TYPES
// ============================================================

enum class TokenizerType {
    CHAR_LEVEL,     // Character-level tokenization
    WORD_LEVEL,     // Word-level tokenization
    BPE,            // Byte Pair Encoding
    WORDPIECE,      // WordPiece (used by BERT)
    SENTENCEPIECE,  // SentencePiece (used by many models)
    UNIGRAM         // Unigram tokenization
};

// ============================================================
//  PARALLELISM SETTINGS
// ============================================================

// Number of threads for CPU parallelism
#ifdef SKULL_NUM_THREADS
    #define SKULL_DEFAULT_THREADS SKULL_NUM_THREADS
#else
    #define SKULL_DEFAULT_THREADS 0  // 0 = auto-detect
#endif

// Enable multi-threading
#define SKULL_MULTI_THREADING 1

// ============================================================
//  MEMORY MANAGEMENT
// ============================================================

// Enable memory pooling for better performance
#define SKULL_MEMORY_POOLING 1

// Maximum memory pool size in MB
#define SKULL_MAX_MEMORY_POOL_MB 1024

// ============================================================
//  DEBUG SETTINGS
// ============================================================

// Enable debug mode
#ifdef SKULL_DEBUG
    #define SKULL_DEBUG_MODE 1
#else
    #define SKULL_DEBUG_MODE 0
#endif

// Enable verbose logging
#ifdef SKULL_VERBOSE
    #define SKULL_VERBOSE_MODE 1
#else
    #define SKULL_VERBOSE_MODE 0
#endif

// Enable profiling
#ifdef SKULL_PROFILE
    #define SKULL_PROFILE_MODE 1
#else
    #define SKULL_PROFILE_MODE 0
#endif

// ============================================================
//  COMPATIBILITY MACROS
// ============================================================

// For Windows
#if SKULL_PLATFORM_WINDOWS
    #define SKULL_PATH_SEPARATOR "\\"
    #define SKULL_LINE_ENDING "\r\n"
    #ifdef SKULL_EXPORTS
        #define SKULL_API __declspec(dllexport)
    #else
        #define SKULL_API __declspec(dllimport)
    #endif
#else
    #define SKULL_PATH_SEPARATOR "/"
    #define SKULL_LINE_ENDING "\n"
    #define SKULL_API
#endif

// For C++17 compatibility
#if __cplusplus < 201703L
    #error "Skull requires C++17 or later"
#endif

// For alignment
#define SKULL_ALIGN(alignment) __attribute__((aligned(alignment)))

// ============================================================
//  UTILITY MACROS
// ============================================================

// Stringify macro
#define SKULL_STRINGIFY(x) #x
#define SKULL_TOSTRING(x) SKULL_STRINGIFY(x)

// Concatenate macros
#define SKULL_CONCAT(a, b) a##b

// Likely/Unlikely hints
#if SKULL_COMPILER_GCC || SKULL_COMPILER_CLANG
    #define SKULL_LIKELY(x) __builtin_expect(!!(x), 1)
    #define SKULL_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define SKULL_LIKELY(x) (x)
    #define SKULL_UNLIKELY(x) (x)
#endif

// No copy/No move macros
#define SKULL_NO_COPY(Type) \
    Type(const Type&) = delete; \
    Type& operator=(const Type&) = delete

#define SKULL_NO_MOVE(Type) \
    Type(Type&&) = delete; \
    Type& operator=(Type&&) = delete

#define SKULL_NO_COPY_NO_MOVE(Type) \
    SKULL_NO_COPY(Type); \
    SKULL_NO_MOVE(Type)

// ============================================================
//  INITIALIZATION
// ============================================================

// Initialize default values
inline void skull_init_config() {
    static bool initialized = false;
    if (initialized) return;
    
    g_precision_mode = PrecisionMode::FP32;
    g_gpu_backend = GPUBackend::AUTO;
    
    initialized = true;
}

// Get Skull version string
inline const char* skull_get_version() {
    return SKULL_VERSION_STRING;
}

// Get build information
inline std::string skull_get_build_info() {
    std::string info = "Skull v" SKULL_VERSION_STRING " (";
    
    #if SKULL_PLATFORM_WINDOWS
        info += "Windows";
    #elif SKULL_PLATFORM_LINUX
        info += "Linux";
    #elif SKULL_PLATFORM_MACOS
        info += "macOS";
    #endif
    
    info += ", ";
    
    #if SKULL_ARCH_X86_64
        info += "x86_64";
    #elif SKULL_ARCH_ARM
        info += "ARM64";
    #endif
    
    info += ") [";
    
    #if SKULL_AVX2
        info += "AVX2";
    #endif
    
    #if SKULL_GPU_OPENCL
        info += ", OpenCL";
    #endif
    
    #if SKULL_GPU_CUDA
        info += ", CUDA";
    #endif
    
    #if SKULL_FP16
        info += ", FP16";
    #endif
    
    info += "]";
    
    return info;
}
