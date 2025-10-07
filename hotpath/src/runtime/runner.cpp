#include "hotpath/hp.h"

#include <iostream>

int main(int, char**) {
    hp_ctx* ctx = nullptr;
    const hp_status ctx_status = hp_ctx_create(nullptr, &ctx);
    if (ctx_status != HP_STATUS_SUCCESS || ctx == nullptr) {
        std::cerr << "hp_ctx_create failed with status " << static_cast<int>(ctx_status) << "\n";
        return 1;
    }

    const hp_status run_status = hp_runner_run(ctx, nullptr);
    if (run_status != HP_STATUS_SUCCESS) {
        std::cerr << "hp_runner_run failed with status " << static_cast<int>(run_status) << "\n";
        hp_ctx_release(ctx);
        return 2;
    }

    hp_ctx_release(ctx);
    return 0;
}

