#include "hp_internal.hpp"
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace {

// Simple hash function for multi-resolution hash encoding
inline uint32_t hash_coord(int32_t x, int32_t y, int32_t z, uint32_t table_size) {
    constexpr uint32_t prime1 = 1u;
    constexpr uint32_t prime2 = 2654435761u;
    constexpr uint32_t prime3 = 805459861u;

    uint32_t hash = (static_cast<uint32_t>(x) * prime1) ^
                    (static_cast<uint32_t>(y) * prime2) ^
                    (static_cast<uint32_t>(z) * prime3);
    return hash % table_size;
}

// Simple multi-resolution hash grid encoding (CPU reference)
void encode_hash_grid_cpu(
    const float pos[3],
    const float* hash_table,
    uint32_t n_levels,
    uint32_t features_per_level,
    uint32_t table_size,
    float base_resolution,
    float finest_resolution,
    float* output
) {
    const float log_scale = std::log(finest_resolution / base_resolution) / static_cast<float>(n_levels > 1 ? n_levels - 1 : 1);

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

// Simple MLP forward pass (CPU reference)
float mlp_forward_sigma_cpu(
    const float* encoded_features,
    const float* weights,
    const float* biases,
    uint32_t input_dim,
    uint32_t hidden_dim
) {
    // First layer: input -> hidden
    float hidden[64];  // Fixed max hidden size
    for (uint32_t i = 0; i < hidden_dim && i < 64; ++i) {
        float sum = biases[i];
        for (uint32_t j = 0; j < input_dim; ++j) {
            sum += encoded_features[j] * weights[i * input_dim + j];
        }
        hidden[i] = std::max(0.0f, sum);  // ReLU activation
    }

    // Second layer: hidden -> output (1D for sigma)
    float output = biases[hidden_dim];
    for (uint32_t i = 0; i < hidden_dim && i < 64; ++i) {
        output += hidden[i] * weights[hidden_dim * input_dim + i];
    }

    return std::max(0.0f, output);  // Ensure non-negative sigma
}

void mlp_forward_color_cpu(
    const float* encoded_features,
    const float* weights,
    const float* biases,
    uint32_t input_dim,
    uint32_t hidden_dim,
    float output_rgb[3]
) {
    // First layer: input -> hidden
    float hidden[64];
    for (uint32_t i = 0; i < hidden_dim && i < 64; ++i) {
        float sum = biases[i];
        for (uint32_t j = 0; j < input_dim; ++j) {
            sum += encoded_features[j] * weights[i * input_dim + j];
        }
        hidden[i] = std::max(0.0f, sum);  // ReLU
    }

    // Second layer: hidden -> output (3D for RGB)
    const uint32_t hidden_output_offset = hidden_dim * input_dim;
    for (int c = 0; c < 3; ++c) {
        float out = biases[hidden_dim + c];
        for (uint32_t i = 0; i < hidden_dim && i < 64; ++i) {
            out += hidden[i] * weights[hidden_output_offset + c * hidden_dim + i];
        }
        output_rgb[c] = std::max(0.0f, std::min(1.0f, out));  // Clamp to [0,1]
    }
}

// Parse hash-MLP parameters from field tensor
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

bool parse_hash_mlp_params(const hp_field* field, HashMLPParams& params) {
    if (field == nullptr || field->source.data == nullptr) {
        return false;
    }

    // For simplicity, use fixed parameters for now
    // In production, these would be stored in the tensor or field metadata
    params.n_levels = 4;
    params.features_per_level = 2;
    params.table_size = 16;  // Small for testing
    params.hidden_dim = 8;
    params.base_resolution = 2.0f;
    params.finest_resolution = 16.0f;

    const float* data = static_cast<const float*>(field->source.data);

    // Layout: hash_table, sigma_weights, sigma_biases, color_weights, color_biases
    const uint32_t encoding_dim = params.n_levels * params.features_per_level;
    const uint32_t hash_table_size = params.n_levels * params.table_size * params.features_per_level;
    const uint32_t sigma_weights_size = params.hidden_dim * encoding_dim + params.hidden_dim;
    const uint32_t sigma_biases_size = params.hidden_dim + 1;
    const uint32_t color_weights_size = params.hidden_dim * encoding_dim + 3 * params.hidden_dim;
    const uint32_t color_biases_size = params.hidden_dim + 3;

    params.hash_table = data;
    params.sigma_weights = data + hash_table_size;
    params.sigma_biases = params.sigma_weights + sigma_weights_size;
    params.color_weights = params.sigma_biases + sigma_biases_size;
    params.color_biases = params.color_weights + color_weights_size;

    return true;
}

}  // namespace

namespace hp_internal {

// CPU reference implementation for hash-MLP sigma sampling
float sample_hash_mlp_sigma_cpu(const hp_field* field, const float pos[3], hp_status* status) {
    if (status != nullptr) {
        *status = HP_STATUS_SUCCESS;
    }

    if (field == nullptr || field->kind != hp_field_kind::hash_mlp) {
        if (status != nullptr) {
            *status = HP_STATUS_INVALID_ARGUMENT;
        }
        return 0.0f;
    }

    HashMLPParams params;
    if (!parse_hash_mlp_params(field, params)) {
        if (status != nullptr) {
            *status = HP_STATUS_INVALID_ARGUMENT;
        }
        return 0.0f;
    }

    // Encode position using multi-resolution hash grid
    float encoded[256];  // Max encoding size
    const uint32_t encoding_dim = params.n_levels * params.features_per_level;

    encode_hash_grid_cpu(pos, params.hash_table, params.n_levels, params.features_per_level,
                         params.table_size, params.base_resolution, params.finest_resolution, encoded);

    // MLP forward pass
    const float sigma = mlp_forward_sigma_cpu(encoded, params.sigma_weights, params.sigma_biases,
                                              encoding_dim, params.hidden_dim);

    return sigma;
}

void sample_hash_mlp_color_cpu(const hp_field* field, const float pos[3], float out_rgb[3], hp_status* status) {
    if (status != nullptr) {
        *status = HP_STATUS_SUCCESS;
    }

    if (out_rgb == nullptr) {
        if (status != nullptr) {
            *status = HP_STATUS_INVALID_ARGUMENT;
        }
        return;
    }

    if (field == nullptr || field->kind != hp_field_kind::hash_mlp) {
        if (status != nullptr) {
            *status = HP_STATUS_INVALID_ARGUMENT;
        }
        out_rgb[0] = 0.0f;
        out_rgb[1] = 0.0f;
        out_rgb[2] = 0.0f;
        return;
    }

    HashMLPParams params;
    if (!parse_hash_mlp_params(field, params)) {
        if (status != nullptr) {
            *status = HP_STATUS_INVALID_ARGUMENT;
        }
        out_rgb[0] = 0.0f;
        out_rgb[1] = 0.0f;
        out_rgb[2] = 0.0f;
        return;
    }

    // Encode position
    float encoded[256];
    const uint32_t encoding_dim = params.n_levels * params.features_per_level;

    encode_hash_grid_cpu(pos, params.hash_table, params.n_levels, params.features_per_level,
                         params.table_size, params.base_resolution, params.finest_resolution, encoded);

    // MLP forward pass
    mlp_forward_color_cpu(encoded, params.color_weights, params.color_biases,
                          encoding_dim, params.hidden_dim, out_rgb);
}

}  // namespace hp_internal
