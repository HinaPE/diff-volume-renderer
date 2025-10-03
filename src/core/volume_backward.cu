#include <cuda_runtime.h>
#include <cstring>
#include "dvren.h"

using namespace dvren;

struct HostCtx
{
    int width;
    int height;
    int n_steps;
    float dt;
    float sigma_scale;
    float stop_thresh;
    FieldProvider field;
};

bool dvren::volume_backward(void* saved_ctx, size_t saved_ctx_bytes, const float* dL_dimage, int width, int height, const FieldProvider& field)
{
    return true;
}
