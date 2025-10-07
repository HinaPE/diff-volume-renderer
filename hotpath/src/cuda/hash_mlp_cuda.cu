#include "hp_internal.hpp"

#if defined(HP_WITH_CUDA)

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace {

// Hash function for multi-resolution hash encoding
__device__ __forceinline__ uint32_t hash_coord(int32_t x, int32_t y, int32_t z, uint32_t table_size) {
    constexpr uint32_t prime1 = 1u;
    constexpr uint32_t prime2 = 2654435761u;
    constexpr uint32_t prime3 = 805459861u;

    uint32_t hash = (static_cast<uint32_t>(x) * prime1) ^
                    (static_cast<uint32_t>(y) * prime2) ^
                    (static_cast<uint32_t>(z) * prime3);
    return hash % table_size;
}

// Multi-resolution hash grid encoding
__device__ void encode_hash_grid(
    const float pos[3],
    const float* hash_table,
    uint32_t n_levels,
    uint32_t features_per_level,
    uint32_t table_size,
    float base_resolution,
    float finest_resolution,
    float* output
) {
    const float log_scale = std::log(finest_resolution / base_resolution) / static_cast<float>(n_levels - 1);

    for (uint32_t level = 0; level < n_levels; ++level) {
        const float resolution = base_resolution * std::exp(static_cast<float>(level) * log_scale);

        // Scale position to current resolution
        const float scaled_pos[3] = {
            pos[0] * resolution,
            pos[1] * resolution,
            pos[2] * resolution
        };

        // Get integer coordinates
        const int32_t x0 = static_cast<int32_t>(std::floor(scaled_pos[0]));
        const int32_t y0 = static_cast<int32_t>(std::floor(scaled_pos[1]));
        const int32_t z0 = static_cast<int32_t>(std::floor(scaled_pos[2]));

        const int32_t x1 = x0 + 1;
        const int32_t y1 = y0 + 1;
        const int32_t z1 = z0 + 1;

        // Fractional parts
        const float fx = scaled_pos[0] - static_cast<float>(x0);
        const float fy = scaled_pos[1] - static_cast<float>(y0);
        const float fz = scaled_pos[2] - static_cast<float>(z0);

        // Trilinear interpolation for each feature
        for (uint32_t feat = 0; feat < features_per_level; ++feat) {
            const uint32_t level_offset = level * table_size * features_per_level;

            // Hash each corner and fetch feature value
            const uint32_t idx000 = hash_coord(x0, y0, z0, table_size);
            const uint32_t idx001 = hash_coord(x0, y0, z1, table_size);
            const uint32_t idx010 = hash_coord(x0, y1, z0, table_size);
            const uint32_t idx011 = hash_coord(x0, y1, z1, table_size);
            const uint32_t idx100 = hash_coord(x1, y0, z0, table_size);
            const uint32_t idx101 = hash_coord(x1, y0, z1, table_size);
            const uint32_t idx110 = hash_coord(x1, y1, z0, table_size);
            const uint32_t idx111 = hash_coord(x1, y1, z1, table_size);

            const float v000 = hash_table[level_offset + idx000 * features_per_level + feat];
            const float v001 = hash_table[level_offset + idx001 * features_per_level + feat];
            const float v010 = hash_table[level_offset + idx010 * features_per_level + feat];
            const float v011 = hash_table[level_offset + idx011 * features_per_level + feat];
            const float v100 = hash_table[level_offset + idx100 * features_per_level + feat];
            const float v101 = hash_table[level_offset + idx101 * features_per_level + feat];
            const float v110 = hash_table[level_offset + idx110 * features_per_level + feat];
            const float v111 = hash_table[level_offset + idx111 * features_per_level + feat];

            // Trilinear interpolation
            const float v00 = v000 * (1.0f - fx) + v100 * fx;
            const float v01 = v001 * (1.0f - fx) + v101 * fx;
            const float v10 = v010 * (1.0f - fx) + v110 * fx;
            const float v11 = v011 * (1.0f - fx) + v111 * fx;

            const float v0 = v00 * (1.0f - fy) + v10 * fy;
            const float v1 = v01 * (1.0f - fy) + v11 * fy;

            output[level * features_per_level + feat] = v0 * (1.0f - fz) + v1 * fz;
        }
    }
}

// Simple MLP forward pass
__device__ float mlp_forward_sigma(
    const float* encoded_features,
    const float* weights,
    const float* biases,
    uint32_t input_dim,
    uint32_t hidden_dim
) {
    // First layer: input -> hidden
    float hidden[64];  // Fixed max hidden size
    for (uint32_t i = 0; i < hidden_dim; ++i) {
        float sum = biases[i];
        for (uint32_t j = 0; j < input_dim; ++j) {
            sum += encoded_features[j] * weights[i * input_dim + j];
        }
        hidden[i] = fmaxf(0.0f, sum);  // ReLU activation
    }

    // Second layer: hidden -> output (1D for sigma)
    float output = biases[hidden_dim];
    for (uint32_t i = 0; i < hidden_dim; ++i) {
        output += hidden[i] * weights[hidden_dim * input_dim + i];
    }

    return fmaxf(0.0f, output);  // Ensure non-negative sigma
}

__device__ void mlp_forward_color(
    const float* encoded_features,
    const float* weights,
    const float* biases,
    uint32_t input_dim,
    uint32_t hidden_dim,
    float output_rgb[3]
) {
    // First layer: input -> hidden
    float hidden[64];
    for (uint32_t i = 0; i < hidden_dim; ++i) {
        float sum = biases[i];
        for (uint32_t j = 0; j < input_dim; ++j) {
            sum += encoded_features[j] * weights[i * input_dim + j];
        }
        hidden[i] = fmaxf(0.0f, sum);  // ReLU
    }

    // Second layer: hidden -> output (3D for RGB)
    const uint32_t hidden_output_offset = hidden_dim * input_dim;
    for (int c = 0; c < 3; ++c) {
        float out = biases[hidden_dim + c];
        for (uint32_t i = 0; i < hidden_dim; ++i) {
            out += hidden[i] * weights[hidden_output_offset + c * hidden_dim + i];
        }
        output_rgb[c] = fmaxf(0.0f, fminf(1.0f, out));  // Clamp to [0,1]
    }
}

struct HashMLPParams {
    const float* hash_table;
    const float* sigma_weights;
    const float* sigma_biases;
    const float* color_weights;
    const float* color_biases;
    uint32_t n_levels;
    uint32_t features_per_level;
    uint32_t table_size;
    uint32_t hidden_dim;
    float base_resolution;
    float finest_resolution;
};

__global__ void sample_hash_mlp_sigma_kernel(
    const float* positions,
    const HashMLPParams params,
    float* output,
    uint32_t n_samples
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    const float pos[3] = {
        positions[idx * 3 + 0],
        positions[idx * 3 + 1],
        positions[idx * 3 + 2]
    };

    // Encode position using multi-resolution hash grid
    float encoded[256];  // Max encoding size
    const uint32_t encoding_dim = params.n_levels * params.features_per_level;
    encode_hash_grid(pos, params.hash_table, params.n_levels, params.features_per_level,
                     params.table_size, params.base_resolution, params.finest_resolution, encoded);

    // MLP forward pass
    output[idx] = mlp_forward_sigma(encoded, params.sigma_weights, params.sigma_biases,
                                     encoding_dim, params.hidden_dim);
}

__global__ void sample_hash_mlp_color_kernel(
    const float* positions,
    const HashMLPParams params,
    float* output,
    uint32_t n_samples
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    const float pos[3] = {
        positions[idx * 3 + 0],
        positions[idx * 3 + 1],
        positions[idx * 3 + 2]
    };

    // Encode position
    float encoded[256];
    const uint32_t encoding_dim = params.n_levels * params.features_per_level;
    encode_hash_grid(pos, params.hash_table, params.n_levels, params.features_per_level,
                     params.table_size, params.base_resolution, params.finest_resolution, encoded);

    // MLP forward pass
    float rgb[3];
    mlp_forward_color(encoded, params.color_weights, params.color_biases,
                      encoding_dim, params.hidden_dim, rgb);

    output[idx * 3 + 0] = rgb[0];
    output[idx * 3 + 1] = rgb[1];
    output[idx * 3 + 2] = rgb[2];
}

}  // namespace

namespace hp_internal {

#if defined(HP_WITH_CUDA)

// CUDA wrapper functions for hash-MLP sampling
hp_status sample_hash_mlp_sigma_cuda(
    const hp_field* field,
    const float* positions_device,
    float* output_device,
    uint32_t n_samples
) {
    if (field == nullptr || positions_device == nullptr || output_device == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    // Parse parameters (in production, this would be cached in the field)
    HashMLPParams params;
    params.n_levels = 4;
    params.features_per_level = 2;
    params.table_size = 16;
    params.hidden_dim = 8;
    params.base_resolution = 2.0f;
    params.finest_resolution = 16.0f;

    // For now, assume params are on device (in production, copy to device if needed)
    const float* data = static_cast<const float*>(field->source.data);
    params.hash_table = data;

    const uint32_t encoding_dim = params.n_levels * params.features_per_level;
    const uint32_t hash_table_size = params.n_levels * params.table_size * params.features_per_level;
    const uint32_t sigma_weights_size = params.hidden_dim * encoding_dim + params.hidden_dim;
    const uint32_t sigma_biases_size = params.hidden_dim + 1;

    params.sigma_weights = data + hash_table_size;
    params.sigma_biases = params.sigma_weights + sigma_weights_size;

    // Launch kernel
    const int block_size = 256;
    const int grid_size = (n_samples + block_size - 1) / block_size;

    sample_hash_mlp_sigma_kernel<<<grid_size, block_size>>>(
        positions_device, params, output_device, n_samples
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return HP_STATUS_INTERNAL_ERROR;
    }

    return HP_STATUS_SUCCESS;
}

hp_status sample_hash_mlp_color_cuda(
    const hp_field* field,
    const float* positions_device,
    float* output_device,
    uint32_t n_samples
) {
    if (field == nullptr || positions_device == nullptr || output_device == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    // Parse parameters
    HashMLPParams params;
    params.n_levels = 4;
    params.features_per_level = 2;
    params.table_size = 16;
    params.hidden_dim = 8;
    params.base_resolution = 2.0f;
    params.finest_resolution = 16.0f;

    const float* data = static_cast<const float*>(field->source.data);
    params.hash_table = data;

    const uint32_t encoding_dim = params.n_levels * params.features_per_level;
    const uint32_t hash_table_size = params.n_levels * params.table_size * params.features_per_level;
    const uint32_t sigma_weights_size = params.hidden_dim * encoding_dim + params.hidden_dim;
    const uint32_t sigma_biases_size = params.hidden_dim + 1;
    const uint32_t color_weights_size = params.hidden_dim * encoding_dim + 3 * params.hidden_dim;

    params.sigma_weights = data + hash_table_size;
    params.sigma_biases = params.sigma_weights + sigma_weights_size;
    params.color_weights = params.sigma_biases + sigma_biases_size;
    params.color_biases = params.color_weights + color_weights_size;

    // Launch kernel
    const int block_size = 256;
    const int grid_size = (n_samples + block_size - 1) / block_size;

    sample_hash_mlp_color_kernel<<<grid_size, block_size>>>(
        positions_device, params, output_device, n_samples
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return HP_STATUS_INTERNAL_ERROR;
    }

    return HP_STATUS_SUCCESS;
}

#endif  // HP_WITH_CUDA

}  // namespace hp_internal

#endif  // HP_WITH_CUDA
