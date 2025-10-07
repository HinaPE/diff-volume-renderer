#include "hotpath/hp.h"
#include "../runtime/hp_internal.hpp"

#if defined(HP_WITH_CUDA)
#include <cuda_runtime.h>
#include <cstring>
#include <new>

namespace hp_internal {

struct hp_graph_exec {
    cudaGraph_t graph{nullptr};
    cudaGraphExec_t graph_exec{nullptr};
    cudaStream_t stream{nullptr};
    bool is_captured{false};

    // Cached buffers for graph capture
    hp_rays_t rays{};
    hp_samp_t samp{};
    hp_intl_t intl{};
    hp_img_t img{};
    hp_grads_t grads{};

    // Workspace pointers
    void* ws_ray{nullptr};
    void* ws_fused{nullptr};
    void* ws_img{nullptr};
    void* ws_diff{nullptr};

    size_t ws_ray_bytes{0};
    size_t ws_fused_bytes{0};
    size_t ws_img_bytes{0};
    size_t ws_diff_bytes{0};

    ~hp_graph_exec() {
        if (graph_exec != nullptr) {
            cudaGraphExecDestroy(graph_exec);
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
};

} // namespace hp_internal

extern "C" {

// Create a CUDA graph executor for the full pipeline
HP_API hp_status hp_graph_create(const hp_plan* plan,
                                  const hp_field* fs,
                                  const hp_field* fc,
                                  size_t ws_ray_bytes,
                                  size_t ws_fused_bytes,
                                  size_t ws_img_bytes,
                                  size_t ws_diff_bytes,
                                  void** out_graph_handle) {
    if (plan == nullptr || fs == nullptr || fc == nullptr || out_graph_handle == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    auto* graph_exec = new (std::nothrow) hp_internal::hp_graph_exec();
    if (graph_exec == nullptr) {
        return HP_STATUS_OUT_OF_MEMORY;
    }

    // Create stream for graph capture
    cudaError_t err = cudaStreamCreate(&graph_exec->stream);
    if (err != cudaSuccess) {
        delete graph_exec;
        return HP_STATUS_INTERNAL_ERROR;
    }

    // Allocate workspace memory
    graph_exec->ws_ray_bytes = ws_ray_bytes;
    graph_exec->ws_fused_bytes = ws_fused_bytes;
    graph_exec->ws_img_bytes = ws_img_bytes;
    graph_exec->ws_diff_bytes = ws_diff_bytes;

    if (ws_ray_bytes > 0) {
        err = cudaMalloc(&graph_exec->ws_ray, ws_ray_bytes);
        if (err != cudaSuccess) {
            delete graph_exec;
            return HP_STATUS_OUT_OF_MEMORY;
        }
    }

    if (ws_fused_bytes > 0) {
        err = cudaMalloc(&graph_exec->ws_fused, ws_fused_bytes);
        if (err != cudaSuccess) {
            delete graph_exec;
            return HP_STATUS_OUT_OF_MEMORY;
        }
    }

    if (ws_img_bytes > 0) {
        err = cudaMalloc(&graph_exec->ws_img, ws_img_bytes);
        if (err != cudaSuccess) {
            delete graph_exec;
            return HP_STATUS_OUT_OF_MEMORY;
        }
    }

    if (ws_diff_bytes > 0) {
        err = cudaMalloc(&graph_exec->ws_diff, ws_diff_bytes);
        if (err != cudaSuccess) {
            delete graph_exec;
            return HP_STATUS_OUT_OF_MEMORY;
        }
    }

    *out_graph_handle = graph_exec;
    return HP_STATUS_SUCCESS;
}

// Capture the CUDA graph for f_ray -> f_fused -> f_img -> f_diff pipeline
HP_API hp_status hp_graph_capture(void* graph_handle,
                                   const hp_plan* plan,
                                   const hp_field* fs,
                                   const hp_field* fc,
                                   const hp_tensor* dL_dI) {
    if (graph_handle == nullptr || plan == nullptr || fs == nullptr || fc == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    auto* graph_exec = static_cast<hp_internal::hp_graph_exec*>(graph_handle);

    // Clean up previous graph if exists
    if (graph_exec->graph_exec != nullptr) {
        cudaGraphExecDestroy(graph_exec->graph_exec);
        graph_exec->graph_exec = nullptr;
    }
    if (graph_exec->graph != nullptr) {
        cudaGraphDestroy(graph_exec->graph);
        graph_exec->graph = nullptr;
    }

    // Begin graph capture
    cudaError_t err = cudaStreamBeginCapture(graph_exec->stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        return HP_STATUS_INTERNAL_ERROR;
    }

    // Execute the pipeline on the capturing stream
    // Note: The actual implementation would need to pass the stream to kernel launches
    // For now, we capture in the default stream context

    hp_status status;

    // 1. Ray generation
    status = hp_internal::ray_generate_cuda(plan, nullptr, &graph_exec->rays,
                                           graph_exec->ws_ray, graph_exec->ws_ray_bytes);
    if (status != HP_STATUS_SUCCESS) {
        cudaStreamEndCapture(graph_exec->stream, &graph_exec->graph);
        return status;
    }

    // 2. Fused sampling + integration
    status = hp_internal::samp_int_fused_cuda(plan, fs, fc, &graph_exec->rays,
                                             &graph_exec->samp, &graph_exec->intl,
                                             graph_exec->ws_fused, graph_exec->ws_fused_bytes);
    if (status != HP_STATUS_SUCCESS) {
        cudaStreamEndCapture(graph_exec->stream, &graph_exec->graph);
        return status;
    }

    // 3. Image composition
    status = hp_internal::img_generate_cuda(plan, &graph_exec->intl, &graph_exec->rays,
                                           &graph_exec->img,
                                           graph_exec->ws_img, graph_exec->ws_img_bytes);
    if (status != HP_STATUS_SUCCESS) {
        cudaStreamEndCapture(graph_exec->stream, &graph_exec->graph);
        return status;
    }

    // 4. Backward pass (if dL_dI provided)
    if (dL_dI != nullptr && dL_dI->data != nullptr) {
        status = hp_internal::diff_generate_cuda(plan, dL_dI, &graph_exec->samp,
                                                &graph_exec->intl, &graph_exec->rays,
                                                &graph_exec->grads,
                                                graph_exec->ws_diff, graph_exec->ws_diff_bytes);
        if (status != HP_STATUS_SUCCESS) {
            cudaStreamEndCapture(graph_exec->stream, &graph_exec->graph);
            return status;
        }
    }

    // End capture
    err = cudaStreamEndCapture(graph_exec->stream, &graph_exec->graph);
    if (err != cudaSuccess) {
        return HP_STATUS_INTERNAL_ERROR;
    }

    // Instantiate the graph
    err = cudaGraphInstantiate(&graph_exec->graph_exec, graph_exec->graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        return HP_STATUS_INTERNAL_ERROR;
    }

    graph_exec->is_captured = true;
    return HP_STATUS_SUCCESS;
}

// Execute the captured CUDA graph
HP_API hp_status hp_graph_execute(void* graph_handle,
                                   hp_rays_t* out_rays,
                                   hp_samp_t* out_samp,
                                   hp_intl_t* out_intl,
                                   hp_img_t* out_img,
                                   hp_grads_t* out_grads) {
    if (graph_handle == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    auto* graph_exec = static_cast<hp_internal::hp_graph_exec*>(graph_handle);

    if (!graph_exec->is_captured || graph_exec->graph_exec == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    // Launch the graph
    cudaError_t err = cudaGraphLaunch(graph_exec->graph_exec, graph_exec->stream);
    if (err != cudaSuccess) {
        return HP_STATUS_INTERNAL_ERROR;
    }

    // Synchronize
    err = cudaStreamSynchronize(graph_exec->stream);
    if (err != cudaSuccess) {
        return HP_STATUS_INTERNAL_ERROR;
    }

    // Copy output pointers
    if (out_rays != nullptr) {
        *out_rays = graph_exec->rays;
    }
    if (out_samp != nullptr) {
        *out_samp = graph_exec->samp;
    }
    if (out_intl != nullptr) {
        *out_intl = graph_exec->intl;
    }
    if (out_img != nullptr) {
        *out_img = graph_exec->img;
    }
    if (out_grads != nullptr) {
        *out_grads = graph_exec->grads;
    }

    return HP_STATUS_SUCCESS;
}

// Destroy the CUDA graph executor
HP_API void hp_graph_release(void* graph_handle) {
    if (graph_handle != nullptr) {
        auto* graph_exec = static_cast<hp_internal::hp_graph_exec*>(graph_handle);

        // Free workspace memory
        if (graph_exec->ws_ray != nullptr) {
            cudaFree(graph_exec->ws_ray);
        }
        if (graph_exec->ws_fused != nullptr) {
            cudaFree(graph_exec->ws_fused);
        }
        if (graph_exec->ws_img != nullptr) {
            cudaFree(graph_exec->ws_img);
        }
        if (graph_exec->ws_diff != nullptr) {
            cudaFree(graph_exec->ws_diff);
        }

        delete graph_exec;
    }
}

} // extern "C"

#endif // HP_WITH_CUDA

