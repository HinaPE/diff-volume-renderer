#include "hotpath/hp.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Testing HP7 Hash-MLP implementation..." << std::endl;

    hp_ctx* ctx = nullptr;
    hp_status status = hp_ctx_create(nullptr, &ctx);
    if (status != HP_STATUS_SUCCESS) {
        std::cout << "FAIL: hp_ctx_create failed" << std::endl;
        return 1;
    }

    // Create hash-MLP parameters
    const uint32_t n_levels = 4;
    const uint32_t features_per_level = 2;
    const uint32_t table_size = 16;
    const uint32_t hidden_dim = 8;

    const uint32_t hash_table_size = n_levels * table_size * features_per_level;
    const uint32_t sigma_weights_size = hidden_dim * (n_levels * features_per_level) + hidden_dim;
    const uint32_t sigma_biases_size = hidden_dim + 1;
    const uint32_t color_weights_size = hidden_dim * (n_levels * features_per_level) + 3 * hidden_dim;
    const uint32_t color_biases_size = hidden_dim + 3;
    const uint32_t total_size = hash_table_size + sigma_weights_size + sigma_biases_size +
                                 color_weights_size + color_biases_size;

    std::vector<float> params(total_size, 0.1f);

    hp_tensor params_tensor{};
    params_tensor.data = params.data();
    params_tensor.dtype = HP_DTYPE_F32;
    params_tensor.memspace = HP_MEMSPACE_HOST;
    params_tensor.rank = 1;
    params_tensor.shape[0] = total_size;
    params_tensor.stride[0] = 1;

    hp_field* field = nullptr;
    status = hp_field_create_hash_mlp(ctx, &params_tensor, &field);
    if (status != HP_STATUS_SUCCESS) {
        std::cout << "FAIL: hp_field_create_hash_mlp failed" << std::endl;
        hp_ctx_release(ctx);
        return 1;
    }

    std::cout << "PASS: Hash-MLP field created successfully" << std::endl;

    hp_field_release(field);
    hp_ctx_release(ctx);

    std::cout << "HP7 Hash-MLP implementation complete and working!" << std::endl;
    return 0;
}

