#pragma once
// ============================================================
//  SKULL PARALLEL PROCESSING v1.0.0
//  Multi-threading and parallel execution support
// ============================================================

#include "tensor.h"
#include "config.h"
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <future>
#include <queue>
#include <functional>
#include <memory>

// ============================================================
//  THREAD POOL
// ============================================================

class ThreadPool {
public:
    ThreadPool(size_t num_threads = SKULL_DEFAULT_THREADS);
    ~ThreadPool();
    
    // Add a task to the queue
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    // Get number of threads
    size_t size() const;
    
    // Wait for all tasks to complete
    void wait();
    
    // Stop the thread pool
    void stop();
    
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop_flag;
    std::atomic<size_t> active_tasks;
};

// ============================================================
//  PARALLEL TENSOR OPERATIONS
// ============================================================

// Parallel matrix multiplication
TensorPtr tensor_matmul_parallel(TensorPtr A, TensorPtr B, size_t num_threads = 0);

// Parallel element-wise operations
TensorPtr tensor_add_parallel(TensorPtr A, TensorPtr B, size_t num_threads = 0);
TensorPtr tensor_mul_parallel(TensorPtr A, TensorPtr B, size_t num_threads = 0);

// Parallel reduction (sum, mean, etc.)
float tensor_sum_parallel(TensorPtr x, size_t num_threads = 0);
float tensor_mean_parallel(TensorPtr x, size_t num_threads = 0);

// Parallel softmax
TensorPtr tensor_softmax_parallel(TensorPtr x, size_t num_threads = 0);

// Parallel ReLU
TensorPtr tensor_relu_parallel(TensorPtr x, size_t num_threads = 0);

// ============================================================
//  THREAD POOL IMPLEMENTATION
// ============================================================

inline ThreadPool::ThreadPool(size_t num_threads)
    : stop_flag(false), active_tasks(0) {
    
    // Auto-detect number of threads if 0
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) {
            num_threads = 4; // Default to 4 threads
        }
    }
    
    // Create worker threads
    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    condition.wait(lock, [this] {
                        return stop_flag || !tasks.empty();
                    });
                    
                    if (stop_flag && tasks.empty()) {
                        return;
                    }
                    
                    task = std::move(tasks.front());
                    tasks.pop();
                    active_tasks++;
                }
                
                task();
                active_tasks--;
            }
        });
    }
}

inline ThreadPool::~ThreadPool() {
    stop();
}

inline void ThreadPool::stop() {
    if (stop_flag) return;
    
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop_flag = true;
    }
    
    condition.notify_all();
    
    for (std::thread& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

inline void ThreadPool::wait() {
    while (active_tasks > 0) {
        std::this_thread::yield();
    }
}

inline size_t ThreadPool::size() const {
    return workers.size();
}

template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    
    using return_type = typename std::result_of<F(Args...)>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    
    std::future<return_type> res = task->get_future();
    
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        tasks.emplace([task]() { (*task)(); });
    }
    
    condition.notify_one();
    
    return res;
}

// ============================================================
//  GLOBAL THREAD POOL
// ============================================================

// Global thread pool instance
inline ThreadPool& get_global_thread_pool() {
    static ThreadPool pool(SKULL_DEFAULT_THREADS);
    return pool;
}

// Set global thread pool size
inline void set_global_thread_pool_size(size_t num_threads) {
    static ThreadPool* pool = nullptr;
    if (pool) {
        pool->stop();
        delete pool;
    }
    pool = new ThreadPool(num_threads);
}

// ============================================================
//  PARALLEL TENSOR OPERATIONS IMPLEMENTATION
// ============================================================

// Helper function to split work into chunks
inline std::vector<std::pair<size_t, size_t>> split_work(size_t total, size_t num_threads) {
    std::vector<std::pair<size_t, size_t>> chunks;
    size_t chunk_size = (total + num_threads - 1) / num_threads;
    
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min((i + 1) * chunk_size, total);
        if (start < end) {
            chunks.emplace_back(start, end);
        }
    }
    
    return chunks;
}

// Parallel matrix multiplication
inline TensorPtr tensor_matmul_parallel(TensorPtr A, TensorPtr B, size_t num_threads) {
    if (num_threads == 0) {
        num_threads = get_global_thread_pool().size();
    }
    
    if (A->cols != B->rows) {
        throw std::runtime_error("Matrix dimensions incompatible for multiplication");
    }
    
    auto result = std::make_shared<Tensor>(A->rows, B->cols);
    
    // Split work by rows
    auto chunks = split_work(A->rows, num_threads);
    
    std::vector<std::future<void>> futures;
    
    for (const auto& chunk : chunks) {
        futures.push_back(get_global_thread_pool().enqueue(
            [A, B, result, chunk]() {
                size_t start_row = chunk.first;
                size_t end_row = chunk.second;
                
                for (size_t i = start_row; i < end_row; ++i) {
                    for (size_t j = 0; j < B->cols; ++j) {
                        float sum = 0.0f;
                        for (size_t k = 0; k < A->cols; ++k) {
                            sum += A->data[i * A->cols + k] * B->data[k * B->cols + j];
                        }
                        result->data[i * result->cols + j] = sum;
                    }
                }
            }));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    return result;
}

// Parallel element-wise addition
inline TensorPtr tensor_add_parallel(TensorPtr A, TensorPtr B, size_t num_threads) {
    if (num_threads == 0) {
        num_threads = get_global_thread_pool().size();
    }
    
    if (A->data.size() != B->data.size()) {
        throw std::runtime_error("Tensor sizes must match for addition");
    }
    
    auto result = std::make_shared<Tensor>(A->rows, A->cols);
    
    // Split work by elements
    auto chunks = split_work(A->data.size(), num_threads);
    
    std::vector<std::future<void>> futures;
    
    for (const auto& chunk : chunks) {
        futures.push_back(get_global_thread_pool().enqueue(
            [A, B, result, chunk]() {
                size_t start = chunk.first;
                size_t end = chunk.second;
                
                for (size_t i = start; i < end; ++i) {
                    result->data[i] = A->data[i] + B->data[i];
                }
            }));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    return result;
}

// Parallel element-wise multiplication
inline TensorPtr tensor_mul_parallel(TensorPtr A, TensorPtr B, size_t num_threads) {
    if (num_threads == 0) {
        num_threads = get_global_thread_pool().size();
    }
    
    if (A->data.size() != B->data.size()) {
        throw std::runtime_error("Tensor sizes must match for element-wise multiplication");
    }
    
    auto result = std::make_shared<Tensor>(A->rows, A->cols);
    
    // Split work by elements
    auto chunks = split_work(A->data.size(), num_threads);
    
    std::vector<std::future<void>> futures;
    
    for (const auto& chunk : chunks) {
        futures.push_back(get_global_thread_pool().enqueue(
            [A, B, result, chunk]() {
                size_t start = chunk.first;
                size_t end = chunk.second;
                
                for (size_t i = start; i < end; ++i) {
                    result->data[i] = A->data[i] * B->data[i];
                }
            }));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    return result;
}

// Parallel sum
inline float tensor_sum_parallel(TensorPtr x, size_t num_threads) {
    if (num_threads == 0) {
        num_threads = get_global_thread_pool().size();
    }
    
    std::vector<float> partial_sums(num_threads, 0.0f);
    
    // Split work by elements
    auto chunks = split_work(x->data.size(), num_threads);
    
    std::vector<std::future<void>> futures;
    
    for (size_t i = 0; i < chunks.size(); ++i) {
        futures.push_back(get_global_thread_pool().enqueue(
            [x, &partial_sums, i, chunks]() {
                size_t start = chunks[i].first;
                size_t end = chunks[i].second;
                
                float sum = 0.0f;
                for (size_t j = start; j < end; ++j) {
                    sum += x->data[j];
                }
                partial_sums[i] = sum;
            }));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    // Sum partial results
    float total = 0.0f;
    for (float sum : partial_sums) {
        total += sum;
    }
    
    return total;
}

// Parallel mean
inline float tensor_mean_parallel(TensorPtr x, size_t num_threads) {
    float sum = tensor_sum_parallel(x, num_threads);
    return sum / x->data.size();
}

// Parallel softmax
inline TensorPtr tensor_softmax_parallel(TensorPtr x, size_t num_threads) {
    if (num_threads == 0) {
        num_threads = get_global_thread_pool().size();
    }
    
    auto result = std::make_shared<Tensor>(x->rows, x->cols);
    
    // Process each row in parallel
    std::vector<std::future<void>> futures;
    
    for (size_t i = 0; i < x->rows; ++i) {
        futures.push_back(get_global_thread_pool().enqueue(
            [x, result, i]() {
                float max_val = -FLT_MAX;
                for (size_t j = 0; j < x->cols; ++j) {
                    if (x->data[i * x->cols + j] > max_val) {
                        max_val = x->data[i * x->cols + j];
                    }
                }
                
                float sum = 0.0f;
                for (size_t j = 0; j < x->cols; ++j) {
                    result->data[i * x->cols + j] = std::exp(x->data[i * x->cols + j] - max_val);
                    sum += result->data[i * x->cols + j];
                }
                
                for (size_t j = 0; j < x->cols; ++j) {
                    result->data[i * x->cols + j] /= sum;
                }
            }));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    return result;
}

// Parallel ReLU
inline TensorPtr tensor_relu_parallel(TensorPtr x, size_t num_threads) {
    if (num_threads == 0) {
        num_threads = get_global_thread_pool().size();
    }
    
    auto result = std::make_shared<Tensor>(x->rows, x->cols);
    
    // Split work by elements
    auto chunks = split_work(x->data.size(), num_threads);
    
    std::vector<std::future<void>> futures;
    
    for (const auto& chunk : chunks) {
        futures.push_back(get_global_thread_pool().enqueue(
            [x, result, chunk]() {
                size_t start = chunk.first;
                size_t end = chunk.second;
                
                for (size_t i = start; i < end; ++i) {
                    result->data[i] = std::max(0.0f, x->data[i]);
                }
            }));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    return result;
}

// ============================================================
//  PARALLEL TRAINING UTILITIES
// ============================================================

// Parallel data loader
class ParallelDataLoader {
public:
    ParallelDataLoader(size_t num_workers = 4, size_t batch_size = 32);
    
    // Load and preprocess data in parallel
    std::vector<TensorPtr> load_batch(const std::vector<std::string>& file_paths);
    
    // Set preprocessing function
    void set_preprocess(std::function<TensorPtr(const std::string&)> preprocess_fn);
    
private:
    size_t num_workers;
    size_t batch_size;
    ThreadPool pool;
    std::function<TensorPtr(const std::string&)> preprocess_fn;
};

inline ParallelDataLoader::ParallelDataLoader(size_t num_workers, size_t batch_size)
    : num_workers(num_workers), batch_size(batch_size), pool(num_workers) {}

inline void ParallelDataLoader::set_preprocess(
    std::function<TensorPtr(const std::string&)> preprocess_fn) {
    this->preprocess_fn = preprocess_fn;
}

inline std::vector<TensorPtr> ParallelDataLoader::load_batch(
    const std::vector<std::string>& file_paths) {
    
    std::vector<TensorPtr> results;
    std::vector<std::future<TensorPtr>> futures;
    
    for (const auto& path : file_paths) {
        futures.push_back(pool.enqueue([this, path]() {
            if (preprocess_fn) {
                return preprocess_fn(path);
            } else {
                // Default: load as text and convert to tensor
                // This is a placeholder - actual implementation would depend on data format
                return tensor_zeros(1, 1);
            }
        }));
    }
    
    // Collect results
    for (auto& future : futures) {
        results.push_back(future.get());
    }
    
    return results;
}

// ============================================================
//  PARALLEL MODEL EVALUATION
// ============================================================

// Evaluate multiple models in parallel
class ParallelEvaluator {
public:
    ParallelEvaluator(size_t num_workers = 4);
    
    // Evaluate multiple models on the same input
    std::vector<TensorPtr> evaluate_models(
        const std::vector<LayerPtr>& models,
        TensorPtr input);
    
    // Evaluate one model on multiple inputs
    std::vector<TensorPtr> evaluate_inputs(
        LayerPtr model,
        const std::vector<TensorPtr>& inputs);
    
private:
    ThreadPool pool;
};

inline ParallelEvaluator::ParallelEvaluator(size_t num_workers)
    : pool(num_workers) {}

inline std::vector<TensorPtr> ParallelEvaluator::evaluate_models(
    const std::vector<LayerPtr>& models,
    TensorPtr input) {
    
    std::vector<TensorPtr> results;
    std::vector<std::future<TensorPtr>> futures;
    
    for (const auto& model : models) {
        futures.push_back(pool.enqueue([model, input]() {
            return model->forward(input);
        }));
    }
    
    // Collect results
    for (auto& future : futures) {
        results.push_back(future.get());
    }
    
    return results;
}

inline std::vector<TensorPtr> ParallelEvaluator::evaluate_inputs(
    LayerPtr model,
    const std::vector<TensorPtr>& inputs) {
    
    std::vector<TensorPtr> results;
    std::vector<std::future<TensorPtr>> futures;
    
    for (const auto& input : inputs) {
        futures.push_back(pool.enqueue([model, input]() {
            return model->forward(input);
        }));
    }
    
    // Collect results
    for (auto& future : futures) {
        results.push_back(future.get());
    }
    
    return results;
}

// ============================================================
//  ASYNC TASK EXECUTION
// ============================================================

// Async task executor for non-blocking operations
class AsyncExecutor {
public:
    AsyncExecutor(size_t max_concurrent_tasks = 8);
    ~AsyncExecutor();
    
    // Execute a task asynchronously
    template<typename F, typename... Args>
    std::future<typename std::result_of<F(Args...)>::type> 
    execute(F&& f, Args&&... args);
    
    // Wait for all tasks to complete
    void wait_all();
    
    // Get number of pending tasks
    size_t pending_tasks() const;
    
private:
    ThreadPool pool;
    std::vector<std::future<void>> pending;
    std::mutex mutex;
};

inline AsyncExecutor::AsyncExecutor(size_t max_concurrent_tasks)
    : pool(max_concurrent_tasks) {}

inline AsyncExecutor::~AsyncExecutor() {
    wait_all();
}

template<typename F, typename... Args>
std::future<typename std::result_of<F(Args...)>::type> 
AsyncExecutor::execute(F&& f, Args&&... args) {
    using return_type = typename std::result_of<F(Args...)>::type;
    
    auto future = pool.enqueue(std::forward<F>(f), std::forward<Args>(args)...);
    
    {
        std::lock_guard<std::mutex> lock(mutex);
        pending.push_back(std::async(std::launch::deferred, [future]() mutable {
            future.wait();
        }));
    }
    
    return future;
}

inline void AsyncExecutor::wait_all() {
    std::vector<std::future<void>> local_pending;
    
    {
        std::lock_guard<std::mutex> lock(mutex);
        local_pending = std::move(pending);
        pending.clear();
    }
    
    for (auto& future : local_pending) {
        future.wait();
    }
    
    pool.wait();
}

inline size_t AsyncExecutor::pending_tasks() const {
    std::lock_guard<std::mutex> lock(mutex);
    return pending.size();
}

// ============================================================
//  PARALLEL UTILITIES
// ============================================================

// Parallel for loop
inline void parallel_for(size_t start, size_t end, 
                         std::function<void(size_t)> func,
                         size_t num_threads = 0) {
    if (num_threads == 0) {
        num_threads = get_global_thread_pool().size();
    }
    
    if (num_threads == 1 || end - start < 1000) {
        // Single-threaded for small work
        for (size_t i = start; i < end; ++i) {
            func(i);
        }
        return;
    }
    
    std::vector<std::future<void>> futures;
    auto chunks = split_work(end - start, num_threads);
    
    for (const auto& chunk : chunks) {
        futures.push_back(get_global_thread_pool().enqueue(
            [func, start, chunk]() {
                for (size_t i = start + chunk.first; i < start + chunk.second; ++i) {
                    func(i);
                }
            }));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
}

// Parallel transform
inline void parallel_transform(const std::vector<float>& input,
                               std::vector<float>& output,
                               std::function<float(float)> func,
                               size_t num_threads = 0) {
    if (num_threads == 0) {
        num_threads = get_global_thread_pool().size();
    }
    
    if (input.size() != output.size()) {
        throw std::runtime_error("Input and output sizes must match");
    }
    
    if (num_threads == 1 || input.size() < 1000) {
        // Single-threaded for small work
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = func(input[i]);
        }
        return;
    }
    
    std::vector<std::future<void>> futures;
    auto chunks = split_work(input.size(), num_threads);
    
    for (const auto& chunk : chunks) {
        futures.push_back(get_global_thread_pool().enqueue(
            [&input, &output, func, chunk]() {
                for (size_t i = chunk.first; i < chunk.second; ++i) {
                    output[i] = func(input[i]);
                }
            }));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
}

// ============================================================
//  INITIALIZATION
// ============================================================

// Initialize parallel processing
inline void skull_init_parallel(size_t num_threads = 0) {
    if (num_threads > 0) {
        set_global_thread_pool_size(num_threads);
    }
    
    // Set global thread pool size based on system
    if (num_threads == 0) {
        unsigned int concurrency = std::thread::hardware_concurrency();
        if (concurrency > 0) {
            set_global_thread_pool_size(concurrency);
        }
    }
}

// Get number of available threads
inline size_t skull_get_num_threads() {
    return get_global_thread_pool().size();
}
