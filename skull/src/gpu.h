#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <memory>
#include <stdexcept>

// ============================================================
//  SKULL GPU  v0.7.0
//  OpenCL-Backend fuer NVIDIA + AMD + Intel GPUs
//
//  OpenCL ist der einzige Standard der auf ALLEN GPUs laeuft:
//    - NVIDIA  (GeForce, RTX, Tesla)
//    - AMD     (RX, Radeon Pro, Instinct)
//    - Intel   (Arc, integrierte GPUs)
//    - Apple   (Metal via OpenCL)
//
//  Strategie:
//    1. OpenCL verfuegbar? -> GPU-Pfad
//    2. Nicht verfuegbar?  -> CPU-Fallback (SIMD)
//
//  GPU-Operationen (Kernel in OpenCL C):
//    - Matrixmultiplikation (wichtigste Operation)
//    - Element-wise relu, sigmoid, tanh
//    - SGD Update
//    - Vektor-Addition/Subtraktion
// ============================================================

// OpenCL einbinden (falls installiert)
#ifdef SKULL_USE_OPENCL
    #ifdef __APPLE__
        #include <OpenCL/opencl.h>
    #else
        #include <CL/cl.h>
    #endif
    #define SKULL_GPU_AVAILABLE 1
#else
    #define SKULL_GPU_AVAILABLE 0
#endif

// ============================================================
//  GPU-STATUS ANZEIGEN
// ============================================================
inline void skull_print_gpu_status() {
#if SKULL_GPU_AVAILABLE
    std::cout << "[Skull GPU] OpenCL geladen\n";

    // Verfuegbare Plattformen und Geraete auflisten
    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        std::cout << "[Skull GPU] Keine OpenCL-Plattformen gefunden\n";
        std::cout << "[Skull GPU] Fallback: CPU (SIMD) wird verwendet\n";
        return;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    for (cl_uint p = 0; p < num_platforms; ++p) {
        char platform_name[256] = {};
        clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME,
                          sizeof(platform_name), platform_name, nullptr);
        std::cout << "[Skull GPU] Plattform: " << platform_name << "\n";

        cl_uint num_devices = 0;
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                       0, nullptr, &num_devices);
        if (num_devices == 0) continue;

        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                       num_devices, devices.data(), nullptr);

        for (cl_uint d = 0; d < num_devices; ++d) {
            char device_name[256] = {};
            cl_ulong global_mem = 0;
            cl_uint compute_units = 0;

            clGetDeviceInfo(devices[d], CL_DEVICE_NAME,
                            sizeof(device_name), device_name, nullptr);
            clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE,
                            sizeof(global_mem), &global_mem, nullptr);
            clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(compute_units), &compute_units, nullptr);

            std::cout << "  [GPU] " << device_name
                      << " | " << (global_mem / (1024*1024)) << " MB"
                      << " | " << compute_units << " CUs\n";
        }
    }
#else
    std::cout << "[Skull GPU] OpenCL nicht kompiliert\n";
    std::cout << "[Skull GPU] CPU-Modus (SIMD AVX2) wird verwendet\n";
    std::cout << "[Skull GPU] Fuer GPU-Support: mit SKULL_USE_OPENCL kompilieren\n";
#endif
}

// ============================================================
//  OPENCL KERNEL QUELLCODE
//  Diese Programme laufen direkt auf der GPU
// ============================================================
static const char* SKULL_MATMUL_KERNEL = R"(
// Matrixmultiplikation: C = A * B
// A: [M x K], B: [K x N], C: [M x N]
// Jeder Work-Item berechnet ein Element von C
__kernel void matmul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int K,
    const int N)
{
    int row = get_global_id(0);  // Zeile von C
    int col = get_global_id(1);  // Spalte von C

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}
)";

static const char* SKULL_ELEMENTWISE_KERNEL = R"(
// ReLU: out = max(0, in)
__kernel void relu(__global const float* in, __global float* out, const int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = fmax(0.0f, in[i]);
}

// Sigmoid: out = 1 / (1 + exp(-in))
__kernel void sigmoid(__global const float* in, __global float* out, const int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = 1.0f / (1.0f + exp(-in[i]));
}

// SGD Update: W = W - lr * grad
__kernel void sgd_update(
    __global float* weights,
    __global const float* grads,
    const float lr,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) weights[i] -= lr * grads[i];
}

// Element-wise Addition: out = a + b
__kernel void add(
    __global const float* a,
    __global const float* b,
    __global float* out,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) out[i] = a[i] + b[i];
}
)";

// ============================================================
//  GPU KONTEXT
//  Verwaltet OpenCL-Verbindung und kompilierte Kernel
// ============================================================
#if SKULL_GPU_AVAILABLE

struct SkullGPU {
    cl_platform_id   platform  = nullptr;
    cl_device_id     device    = nullptr;
    cl_context       context   = nullptr;
    cl_command_queue queue     = nullptr;
    cl_program       program   = nullptr;

    cl_kernel k_matmul  = nullptr;
    cl_kernel k_relu    = nullptr;
    cl_kernel k_sigmoid = nullptr;
    cl_kernel k_sgd     = nullptr;
    cl_kernel k_add     = nullptr;

    bool initialized = false;
    std::string device_name;

    // GPU initialisieren — bevorzugt AMD oder NVIDIA
    bool init(bool prefer_amd = false) {
        cl_uint num_platforms = 0;
        if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS
            || num_platforms == 0) {
            std::cerr << "[GPU] Keine OpenCL-Plattformen gefunden\n";
            return false;
        }

        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

        // Beste GPU auswaehlen
        cl_device_id best_device = nullptr;
        cl_platform_id best_platform = nullptr;
        bool found_dedicated = false;

        for (auto& plat : platforms) {
            char pname[256] = {};
            clGetPlatformInfo(plat, CL_PLATFORM_NAME, 256, pname, nullptr);
            std::string pname_str(pname);

            cl_uint num_dev = 0;
            clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_dev);
            if (num_dev == 0) continue;

            std::vector<cl_device_id> devs(num_dev);
            clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, num_dev, devs.data(), nullptr);

            for (auto& dev : devs) {
                char dname[256] = {};
                clGetDeviceInfo(dev, CL_DEVICE_NAME, 256, dname, nullptr);
                std::string dname_str(dname);

                // AMD bevorzugen wenn gewuenscht
                bool is_amd = dname_str.find("AMD") != std::string::npos ||
                              dname_str.find("Radeon") != std::string::npos ||
                              pname_str.find("AMD") != std::string::npos;
                bool is_nvidia = dname_str.find("NVIDIA") != std::string::npos ||
                                 dname_str.find("GeForce") != std::string::npos ||
                                 dname_str.find("RTX") != std::string::npos;

                if (!found_dedicated || (prefer_amd && is_amd) ||
                    (!prefer_amd && is_nvidia)) {
                    best_device   = dev;
                    best_platform = plat;
                    device_name   = dname_str;
                    found_dedicated = true;
                    if ((prefer_amd && is_amd) || (!prefer_amd && is_nvidia))
                        goto found_gpu; // Ideale GPU gefunden
                }
            }
        }

        found_gpu:
        if (!best_device) {
            // Fallback: Erste verfuegbare CPU via OpenCL
            for (auto& plat : platforms) {
                cl_uint num_dev = 0;
                clGetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, 0, nullptr, &num_dev);
                if (num_dev == 0) continue;
                std::vector<cl_device_id> devs(num_dev);
                clGetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, num_dev, devs.data(), nullptr);
                best_device   = devs[0];
                best_platform = plat;
                char dname[256] = {};
                clGetDeviceInfo(best_device, CL_DEVICE_NAME, 256, dname, nullptr);
                device_name = std::string(dname) + " (CPU via OpenCL)";
                break;
            }
        }

        if (!best_device) {
            std::cerr << "[GPU] Kein OpenCL-Geraet gefunden\n";
            return false;
        }

        platform = best_platform;
        device   = best_device;

        // Kontext und Queue erstellen
        cl_int err;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) { std::cerr << "[GPU] Kontext-Fehler\n"; return false; }

        queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) { std::cerr << "[GPU] Queue-Fehler\n"; return false; }

        // Kernel kompilieren
        std::string all_kernels = std::string(SKULL_MATMUL_KERNEL) +
                                  std::string(SKULL_ELEMENTWISE_KERNEL);
        const char* src = all_kernels.c_str();
        size_t src_len  = all_kernels.size();

        program = clCreateProgramWithSource(context, 1, &src, &src_len, &err);
        if (err != CL_SUCCESS) { std::cerr << "[GPU] Programm-Fehler\n"; return false; }

        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            char log[4096] = {};
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                  sizeof(log), log, nullptr);
            std::cerr << "[GPU] Kompilierungsfehler:\n" << log << "\n";
            return false;
        }

        // Kernel-Handles erstellen
        k_matmul  = clCreateKernel(program, "matmul",     &err);
        k_relu    = clCreateKernel(program, "relu",       &err);
        k_sigmoid = clCreateKernel(program, "sigmoid",    &err);
        k_sgd     = clCreateKernel(program, "sgd_update", &err);
        k_add     = clCreateKernel(program, "add",        &err);

        initialized = true;
        std::cout << "[Skull GPU] Initialisiert: " << device_name << "\n";
        return true;
    }

    // GPU-Matrixmultiplikation: C = A * B
    // A: [M x K] (float), B: [K x N] (float), C: [M x N] (float)
    std::vector<float> matmul_gpu(
        const std::vector<float>& A,
        const std::vector<float>& B,
        int M, int K, int N)
    {
        std::vector<float> C(M * N, 0.0f);
        if (!initialized) return C;

        cl_int err;
        // Buffer auf GPU anlegen
        cl_mem buf_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       A.size() * sizeof(float),
                                       const_cast<float*>(A.data()), &err);
        cl_mem buf_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       B.size() * sizeof(float),
                                       const_cast<float*>(B.data()), &err);
        cl_mem buf_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                       C.size() * sizeof(float), nullptr, &err);

        // Kernel-Argumente setzen
        clSetKernelArg(k_matmul, 0, sizeof(cl_mem), &buf_A);
        clSetKernelArg(k_matmul, 1, sizeof(cl_mem), &buf_B);
        clSetKernelArg(k_matmul, 2, sizeof(cl_mem), &buf_C);
        clSetKernelArg(k_matmul, 3, sizeof(int), &M);
        clSetKernelArg(k_matmul, 4, sizeof(int), &K);
        clSetKernelArg(k_matmul, 5, sizeof(int), &N);

        // Kernel ausfuehren (2D: jede Zelle von C ist ein Work-Item)
        size_t global[2] = { (size_t)M, (size_t)N };
        clEnqueueNDRangeKernel(queue, k_matmul, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
        clFinish(queue);

        // Ergebnis zurueck zur CPU
        clEnqueueReadBuffer(queue, buf_C, CL_TRUE, 0,
                            C.size() * sizeof(float), C.data(), 0, nullptr, nullptr);

        clReleaseMemObject(buf_A);
        clReleaseMemObject(buf_B);
        clReleaseMemObject(buf_C);
        return C;
    }

    // SGD Update auf GPU
    void sgd_update_gpu(std::vector<float>& weights,
                        const std::vector<float>& grads,
                        float lr)
    {
        if (!initialized) return;
        int n = (int)weights.size();
        cl_int err;
        cl_mem buf_w = clCreateBuffer(context,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            n * sizeof(float), weights.data(), &err);
        cl_mem buf_g = clCreateBuffer(context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            n * sizeof(float), const_cast<float*>(grads.data()), &err);

        clSetKernelArg(k_sgd, 0, sizeof(cl_mem), &buf_w);
        clSetKernelArg(k_sgd, 1, sizeof(cl_mem), &buf_g);
        clSetKernelArg(k_sgd, 2, sizeof(float),  &lr);
        clSetKernelArg(k_sgd, 3, sizeof(int),    &n);

        size_t global = (size_t)n;
        clEnqueueNDRangeKernel(queue, k_sgd, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        clFinish(queue);
        clEnqueueReadBuffer(queue, buf_w, CL_TRUE, 0,
                            n * sizeof(float), weights.data(), 0, nullptr, nullptr);
        clReleaseMemObject(buf_w);
        clReleaseMemObject(buf_g);
    }

    ~SkullGPU() {
        if (k_matmul)  clReleaseKernel(k_matmul);
        if (k_relu)    clReleaseKernel(k_relu);
        if (k_sigmoid) clReleaseKernel(k_sigmoid);
        if (k_sgd)     clReleaseKernel(k_sgd);
        if (k_add)     clReleaseKernel(k_add);
        if (program)   clReleaseProgram(program);
        if (queue)     clReleaseCommandQueue(queue);
        if (context)   clReleaseContext(context);
    }
};

// Globale GPU-Instanz
inline SkullGPU& skull_gpu() {
    static SkullGPU instance;
    return instance;
}

inline bool skull_init_gpu(bool prefer_amd = false) {
    return skull_gpu().init(prefer_amd);
}

#else // Kein OpenCL

// Fallback-Stubs wenn OpenCL nicht kompiliert
inline bool skull_init_gpu(bool prefer_amd = false) {
    std::cout << "[Skull GPU] OpenCL nicht verfuegbar\n";
    std::cout << "[Skull GPU] Verwende CPU (AVX2 SIMD)\n";
    return false;
}

#endif // SKULL_GPU_AVAILABLE
