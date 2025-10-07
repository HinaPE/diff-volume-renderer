#include "hotpath/hp.h"

#include <iostream>

int main(int, char**) {
    hp_ctx* ctx = nullptr;
    const hp_status ctx_status = hp_ctx_create(nullptr, &ctx);
    if (ctx_status != HP_STATUS_SUCCESS || ctx == nullptr) {
        std::cerr << "hp_ctx_create failed with status " << static_cast<int>(ctx_status) << "\n";
        return 1;
    }

    hp_runner_options opts{};
    opts.manifest_path = "tests/manifest.yaml";
    opts.thresholds_path = "tests/thresholds.yaml";
    opts.perf_scenarios_path = "tests/perf_scenarios.yaml";

    const hp_status run_status = hp_runner_run(ctx, &opts);
    hp_ctx_release(ctx);

    return run_status == HP_STATUS_SUCCESS ? 0 : 1;
}

