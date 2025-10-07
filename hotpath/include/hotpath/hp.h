#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef _MSC_VER
#  ifdef HP_BUILD_DLL
#    define HP_API __declspec(dllexport)
#  else
#    define HP_API
#  endif
#else
#  define HP_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define HP_VERSION_MAJOR 0U
#define HP_VERSION_MINOR 1U
#define HP_VERSION_PATCH 0U

typedef struct hp_version {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
} hp_version;

typedef enum hp_status {
    HP_STATUS_SUCCESS = 0,
    HP_STATUS_INVALID_ARGUMENT = 1,
    HP_STATUS_OUT_OF_MEMORY = 2,
    HP_STATUS_NOT_IMPLEMENTED = 3,
    HP_STATUS_UNSUPPORTED = 4,
    HP_STATUS_INTERNAL_ERROR = 5
} hp_status;

typedef enum hp_memspace {
    HP_MEMSPACE_HOST = 0,
    HP_MEMSPACE_DEVICE = 1
} hp_memspace;

typedef enum hp_dtype {
    HP_DTYPE_F16 = 0,
    HP_DTYPE_BF16 = 1,
    HP_DTYPE_F32 = 2,
    HP_DTYPE_I32 = 3,
    HP_DTYPE_U32 = 4
} hp_dtype;

typedef enum hp_camera_model {
    HP_CAMERA_PINHOLE = 0,
    HP_CAMERA_ORTHOGRAPHIC = 1
} hp_camera_model;

typedef struct hp_tensor {
    void* data;
    hp_dtype dtype;
    hp_memspace memspace;
    uint32_t rank;
    int64_t shape[8];
    int64_t stride[8];
} hp_tensor;

typedef struct hp_ctx_desc {
    uint32_t flags;
    const char* preferred_device;
    const void* reserved;
} hp_ctx_desc;

typedef struct hp_camera_desc {
    hp_camera_model model;
    float K[9];
    float c2w[12];
    float ortho_scale;
} hp_camera_desc;

typedef struct hp_roi_desc {
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
} hp_roi_desc;

typedef struct hp_plan_desc {
    uint32_t width;
    uint32_t height;
    float t_near;
    float t_far;
    uint32_t max_rays;
    uint32_t max_samples;
    uint64_t seed;
    hp_camera_desc camera;
    hp_roi_desc roi;
} hp_plan_desc;

typedef struct hp_ctx hp_ctx;
typedef struct hp_plan hp_plan;
typedef struct hp_field hp_field;

typedef struct hp_rays_t {
    hp_tensor origins;     /* shape: (N, 3) */
    hp_tensor directions;  /* shape: (N, 3), normalized */
    hp_tensor t_near;      /* shape: (N,) */
    hp_tensor t_far;       /* shape: (N,) */
    hp_tensor pixel_ids;   /* shape: (N,) uint32 */
} hp_rays_t;

typedef struct hp_samp_t {
    hp_tensor positions;   /* shape: (M, 3) */
    hp_tensor dt;          /* shape: (M,) */
    hp_tensor ray_offset;  /* shape: (N + 1,) prefix offsets */
    hp_tensor sigma;       /* shape: (M,) */
    hp_tensor color;       /* shape: (M, 3) */
} hp_samp_t;

typedef struct hp_intl_t {
    hp_tensor radiance;    /* shape: (N, 3) */
    hp_tensor transmittance;
    hp_tensor opacity;
    hp_tensor depth;
    hp_tensor aux;
} hp_intl_t;

typedef struct hp_img_t {
    hp_tensor image;       /* shape: (H, W, 3) */
    hp_tensor trans;
    hp_tensor opacity;
    hp_tensor depth;
    hp_tensor hitmask;
} hp_img_t;

typedef struct hp_grads_t {
    hp_tensor sigma;
    hp_tensor color;
    hp_tensor camera;
} hp_grads_t;

typedef struct hp_runner_options {
    const char* manifest_path;
    const char* thresholds_path;
    const char* perf_scenarios_path;
} hp_runner_options;

HP_API hp_version hp_get_version(void);

HP_API hp_status hp_ctx_create(const hp_ctx_desc* desc, hp_ctx** out_ctx);
HP_API void hp_ctx_release(hp_ctx* ctx);
HP_API hp_status hp_ctx_get_desc(const hp_ctx* ctx, hp_ctx_desc* out_desc);

HP_API hp_status hp_plan_create(const hp_ctx* ctx, const hp_plan_desc* desc, hp_plan** out_plan);
HP_API void hp_plan_release(hp_plan* plan);
HP_API hp_status hp_plan_get_desc(const hp_plan* plan, hp_plan_desc* out_desc);

HP_API hp_status hp_ray(const hp_plan* plan, const hp_rays_t* override_or_null, hp_rays_t* rays, void* ws, size_t ws_bytes);
HP_API hp_status hp_samp(const hp_plan* plan, const hp_field* fs, const hp_field* fc, const hp_rays_t* rays, hp_samp_t* samp, void* ws, size_t ws_bytes);
HP_API hp_status hp_int(const hp_plan* plan, const hp_samp_t* samp, hp_intl_t* intl, void* ws, size_t ws_bytes);
HP_API hp_status hp_img(const hp_plan* plan, const hp_intl_t* intl, const hp_rays_t* rays, hp_img_t* img, void* ws, size_t ws_bytes);
HP_API hp_status hp_diff(const hp_plan* plan, const hp_tensor* dL_dI, const hp_samp_t* samp, const hp_intl_t* intl, hp_grads_t* grads, void* ws, size_t ws_bytes);

HP_API hp_status hp_field_create_grid_sigma(const hp_ctx* ctx, const hp_tensor* grid, uint32_t interp, uint32_t oob, hp_field** out_field);
HP_API hp_status hp_field_create_grid_color(const hp_ctx* ctx, const hp_tensor* grid, uint32_t interp, uint32_t oob, hp_field** out_field);
HP_API hp_status hp_field_create_hash_mlp(const hp_ctx* ctx, const hp_tensor* params, hp_field** out_field);
HP_API void hp_field_release(hp_field* field);

HP_API hp_status hp_runner_run(const hp_ctx* ctx, const hp_runner_options* options);

#ifdef __cplusplus
}
#endif
