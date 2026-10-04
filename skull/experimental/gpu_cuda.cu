// ============================================================
//  SKULL CUDA KERNELS v1.0.0
//  High-performance CUDA kernels for NVIDIA GPUs
// ============================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

// ============================================================
//  MEMORY MANAGEMENT
// ============================================================

// CUDA error checking macro
#define CUDA_CHECK(call) {
    cudaError_t err = call;
    if (err != cudaSuccess) {
        printf("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// ============================================================
//  MATRIX MULTIPLICATION KERNELS
// ============================================================

// FP32 Matrix Multiplication (C = A * B)
// A: [M x K], B: [K x N], C: [M x N]
__global__ void matmul_fp32(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// FP16 Matrix Multiplication
__global__ void matmul_fp16(
    const half* A,
    const half* B,
    half* C,
    int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    half sum = __float2half(0.0f);
    for (int k = 0; k < K; ++k) {
        sum = __hadd(sum, __hmul(A[row * K + k], B[k * N + col]));
    }
    C[row * N + col] = sum;
}

// Mixed Precision Matrix Multiplication (FP16 inputs, FP32 accumulation)
__global__ void matmul_mixed(
    const half* A,
    const half* B,
    float* C,
    int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    }
    C[row * N + col] = sum;
}

// Optimized Matrix Multiplication with shared memory
__global__ void matmul_fp32_optimized(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N)
{
    // Shared memory for tiles
    __shared__ float sA[32][32];
    __shared__ float sB[32][32];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + 31) / 32; ++t) {
        // Load tile from A
        if (row < M && (threadIdx.x + t * 32) < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + threadIdx.x + t * 32];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B
        if ((threadIdx.y + t * 32) < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + t * 32) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute tile multiplication
        for (int k = 0; k < 32; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================
//  ACTIVATION FUNCTION KERNELS
// ============================================================

// ReLU
__global__ void relu_fp32(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaxf(0.0f, in[idx]);
    }
}

// Leaky ReLU
__global__ void leaky_relu_fp32(const float* in, float* out, int n, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (in[idx] > 0.0f) ? in[idx] : in[idx] * alpha;
    }
}

// GELU (Gaussian Error Linear Unit)
__global__ void gelu_fp32(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        out[idx] = x * 0.5f * (1.0f + tanhf(inner));
    }
}

// Swish
__global__ void swish_fp32(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        out[idx] = x / (1.0f + expf(-x));
    }
}

// Sigmoid
__global__ void sigmoid_fp32(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f / (1.0f + expf(-in[idx]));
    }
}

// Tanh
__global__ void tanh_fp32(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanhf(in[idx]);
    }
}

// Softmax
__global__ void softmax_fp32(float* x, int n, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Find max for numerical stability
    __shared__ float s_max[256];
    if (threadIdx.x < 256) s_max[threadIdx.x] = -FLT_MAX;
    __syncthreads();
    
    int batch_idx = idx / n;
    int pos = idx % n;
    
    if (idx < batch_size * n) {
        atomicMax(&s_max[batch_idx], x[idx]);
    }
    __syncthreads();
    
    // Subtract max and compute exp
    if (idx < batch_size * n) {
        x[idx] = expf(x[idx] - s_max[batch_idx]);
    }
    __syncthreads();
    
    // Sum all exponentials
    __shared__ float s_sum[256];
    if (threadIdx.x < 256) s_sum[threadIdx.x] = 0.0f;
    __syncthreads();
    
    if (idx < batch_size * n) {
        atomicAdd(&s_sum[batch_idx], x[idx]);
    }
    __syncthreads();
    
    // Normalize
    if (idx < batch_size * n) {
        x[idx] /= s_sum[batch_idx];
    }
}

// ============================================================
//  OPTIMIZER KERNELS
// ============================================================

// SGD Update: W = W - lr * grad
__global__ void sgd_update_fp32(
    float* weights,
    const float* grads,
    float lr,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        weights[idx] -= lr * grads[idx];
    }
}

// Adam Update
__global__ void adam_update_fp32(
    float* weights,
    const float* grads,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int t,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grads[idx];
        
        // Update biased second moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grads[idx] * grads[idx];
        
        // Compute bias-corrected estimates
        float m_hat = m[idx] / (1.0f - powf(beta1, t));
        float v_hat = v[idx] / (1.0f - powf(beta2, t));
        
        // Update weights
        weights[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

// AdamW Update (Adam with weight decay)
__global__ void adamw_update_fp32(
    float* weights,
    const float* grads,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int t,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Weight decay
        float decay = lr * weight_decay;
        
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grads[idx];
        
        // Update biased second moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grads[idx] * grads[idx];
        
        // Compute bias-corrected estimates
        float m_hat = m[idx] / (1.0f - powf(beta1, t));
        float v_hat = v[idx] / (1.0f - powf(beta2, t));
        
        // Update weights with weight decay
        weights[idx] -= decay * weights[idx] + lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

// Lion Update (new efficient optimizer)
__global__ void lion_update_fp32(
    float* weights,
    const float* grads,
    float* momentum,
    float lr,
    float beta1,
    float beta2,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Update momentum
        momentum[idx] = (1.0f - beta1) * grads[idx] + beta1 * momentum[idx];
        
        // Update weights
        float sign = (grads[idx] > 0.0f) ? 1.0f : -1.0f;
        weights[idx] -= lr * (sign + beta2 * momentum[idx]);
    }
}

// ============================================================
//  ATTENTION KERNELS
// ============================================================

// Scaled Dot-Product Attention
__global__ void attention_fp32(
    const float* Q,  // Query [batch, heads, seq_len, d_k]
    const float* K,  // Key [batch, heads, seq_len, d_k]
    const float* V,  // Value [batch, heads, seq_len, d_v]
    float* output,   // Output [batch, heads, seq_len, d_v]
    int batch_size,
    int num_heads,
    int seq_len,
    int d_k,
    int d_v,
    float scale)
{
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = threadIdx.y;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || 
        row >= seq_len || col >= d_v) return;
    
    // Compute attention scores
    __shared__ float s_scores[32][32];
    
    int q_offset = ((batch_idx * num_heads + head_idx) * seq_len + row) * d_k;
    int k_offset = ((batch_idx * num_heads + head_idx) * seq_len) * d_k;
    
    float max_score = -FLT_MAX;
    for (int i = 0; i < seq_len; ++i) {
        float score = 0.0f;
        for (int j = 0; j < d_k; ++j) {
            score += Q[q_offset + j] * K[k_offset + i * d_k + j];
        }
        score *= scale;
        s_scores[threadIdx.x][i] = score;
        if (score > max_score) max_score = score;
    }
    
    __syncthreads();
    
    // Softmax
    float sum = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        s_scores[threadIdx.x][i] = expf(s_scores[threadIdx.x][i] - max_score);
        sum += s_scores[threadIdx.x][i];
    }
    
    for (int i = 0; i < seq_len; ++i) {
        s_scores[threadIdx.x][i] /= sum;
    }
    
    __syncthreads();
    
    // Weighted sum of values
    float result = 0.0f;
    int v_offset = ((batch_idx * num_heads + head_idx) * seq_len) * d_v;
    
    for (int i = 0; i < seq_len; ++i) {
        result += s_scores[threadIdx.x][i] * V[v_offset + i * d_v + col];
    }
    
    int out_offset = ((batch_idx * num_heads + head_idx) * seq_len + row) * d_v + col;
    output[out_offset] = result;
}

// FlashAttention-like kernel (memory efficient)
__global__ void flash_attention_fp32(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int batch_size,
    int num_heads,
    int seq_len,
    int d_k,
    int d_v,
    float scale)
{
    // Implement memory-efficient attention
    // This is a simplified version - full FlashAttention is more complex
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size * num_heads * seq_len * d_v) return;
    
    int batch_idx = idx / (num_heads * seq_len * d_v);
    int head_idx = (idx % (num_heads * seq_len * d_v)) / (seq_len * d_v);
    int row = (idx % (seq_len * d_v)) / d_v;
    int col = idx % d_v;
    
    // Compute attention for this position
    int q_offset = ((batch_idx * num_heads + head_idx) * seq_len + row) * d_k;
    int k_offset = ((batch_idx * num_heads + head_idx) * seq_len) * d_k;
    int v_offset = ((batch_idx * num_heads + head_idx) * seq_len) * d_v;
    
    float max_score = -FLT_MAX;
    float scores[128]; // Max seq_len for shared memory
    
    for (int i = 0; i < seq_len; ++i) {
        float score = 0.0f;
        for (int j = 0; j < d_k; ++j) {
            score += Q[q_offset + j] * K[k_offset + i * d_k + j];
        }
        score *= scale;
        scores[i] = score;
        if (score > max_score) max_score = score;
    }
    
    // Softmax
    float sum = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        scores[i] = expf(scores[i] - max_score);
        sum += scores[i];
    }
    
    for (int i = 0; i < seq_len; ++i) {
        scores[i] /= sum;
    }
    
    // Weighted sum
    float result = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        result += scores[i] * V[v_offset + i * d_v + col];
    }
    
    output[((batch_idx * num_heads + head_idx) * seq_len + row) * d_v + col] = result;
}

// ============================================================
//  LAYER NORMALIZATION
// ============================================================

__global__ void layer_norm_fp32(
    float* x,
    const float* gamma,
    const float* beta,
    int n,
    int batch_size,
    float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int batch_idx = idx / n;
    int pos = idx % n;
    
    if (batch_idx >= batch_size) return;
    
    // Compute mean
    __shared__ float s_mean[256];
    __shared__ float s_var[256];
    
    if (threadIdx.x < 256) {
        s_mean[threadIdx.x] = 0.0f;
        s_var[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    float x_val = x[batch_idx * n + pos];
    atomicAdd(&s_mean[batch_idx], x_val);
    __syncthreads();
    
    if (threadIdx.x == 0 && batch_idx < 256) {
        s_mean[batch_idx] /= n;
    }
    __syncthreads();
    
    // Compute variance
    x_val = x[batch_idx * n + pos] - s_mean[batch_idx];
    atomicAdd(&s_var[batch_idx], x_val * x_val);
    __syncthreads();
    
    if (threadIdx.x == 0 && batch_idx < 256) {
        s_var[batch_idx] = sqrtf(s_var[batch_idx] / n + eps);
    }
    __syncthreads();
    
    // Normalize
    x_val = x[batch_idx * n + pos];
    x[batch_idx * n + pos] = gamma[pos] * (x_val - s_mean[batch_idx]) / s_var[batch_idx] + beta[pos];
}

// ============================================================
//  RESIDUAL CONNECTION
// ============================================================

__global__ void residual_add_fp32(
    float* x,
    const float* residual,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] += residual[idx];
    }
}

// ============================================================
//  DROPOUT
// ============================================================

__global__ void dropout_fp32(
    float* x,
    float* mask,
    float p,
    int n,
    unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple dropout implementation
        float r = rand() / (float)RAND_MAX;
        if (r < p) {
            x[idx] = 0.0f;
            mask[idx] = 0.0f;
        } else {
            x[idx] /= (1.0f - p);
            mask[idx] = 1.0f;
        }
    }
}

// ============================================================
//  UTILITY KERNELS
// ============================================================

// Element-wise addition
__global__ void add_fp32(
    const float* a,
    const float* b,
    float* out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

// Element-wise multiplication
__global__ void mul_fp32(
    const float* a,
    const float* b,
    float* out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

// Copy kernel
__global__ void copy_fp32(
    const float* src,
    float* dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Fill with constant
__global__ void fill_fp32(
    float* arr,
    float value,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = value;
    }
}

// ============================================================
//  CUDA UTILITY FUNCTIONS
// ============================================================

// Initialize CUDA context
bool skull_cuda_init() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        printf("[CUDA] No NVIDIA GPU found\n");
        return false;
    }
    
    // Use device 0 by default
    CUDA_CHECK(cudaSetDevice(0));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("[CUDA] Device: %s\n", prop.name);
    printf("[CUDA] Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("[CUDA] Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
    printf("[CUDA] Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("[CUDA] Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    
    return true;
}

// Get optimal block size for a kernel
int skull_cuda_get_block_size(int max_threads) {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    return std::min(max_threads, prop.maxThreadsPerBlock);
}

// Get optimal grid size
int skull_cuda_get_grid_size(int n, int block_size) {
    return (n + block_size - 1) / block_size;
}
