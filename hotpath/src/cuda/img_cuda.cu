#include "hp_internal.hpp"

#if defined(HP_WITH_CUDA)

#include <cuda_runtime.h>
#include "../runtime/workspace.hpp"
#include <vector>

namespace hp_internal {

namespace {

struct DeviceImgBuffers {
    float* image{nullptr};
    float* trans{nullptr};
    float* opacity{nullptr};
    float* depth{nullptr};
    uint32_t* hitmask{nullptr};
};

hp_status ensure_image_buffers_device(const hp_plan* plan,
                                      hp_img_t* img,
                                      void* ws,
                                      size_t ws_bytes,
                                      DeviceImgBuffers& buffers) {
    const size_t width = plan->desc.width;
    const size_t height = plan->desc.height;
    const size_t pixel_count = width * height;
    const int64_t width_i64 = static_cast<int64_t>(width);
    const int64_t height_i64 = static_cast<int64_t>(height);

    img->image.dtype = HP_DTYPE_F32;
    img->image.memspace = HP_MEMSPACE_DEVICE;
    img->image.rank = 3;
    img->image.shape[0] = height_i64;
    img->image.shape[1] = width_i64;
    img->image.shape[2] = 3;
    img->image.stride[2] = 1;
    img->image.stride[1] = 3;
    img->image.stride[0] = width_i64 * 3;

    img->trans.dtype = HP_DTYPE_F32;
    img->trans.memspace = HP_MEMSPACE_DEVICE;
    img->trans.rank = 2;
    img->trans.shape[0] = height_i64;
    img->trans.shape[1] = width_i64;
    img->trans.stride[1] = 1;
    img->trans.stride[0] = width_i64;

    img->opacity = img->trans;
    img->depth = img->trans;
    img->hitmask.dtype = HP_DTYPE_U32;
    img->hitmask.memspace = HP_MEMSPACE_DEVICE;
    img->hitmask.rank = 2;
    img->hitmask.shape[0] = height_i64;
    img->hitmask.shape[1] = width_i64;
    img->hitmask.stride[1] = 1;
    img->hitmask.stride[0] = width_i64;

    hp_workspace_allocator allocator(ws, ws_bytes);
    const size_t image_bytes = pixel_count * 3U * sizeof(float);
    const size_t scalar_bytes = pixel_count * sizeof(float);
    const size_t mask_bytes = pixel_count * sizeof(uint32_t);

    if (image_bytes > 0 && img->image.data == nullptr) {
        img->image.data = allocator.allocate(image_bytes, alignof(float));
    }
    if (scalar_bytes > 0) {
        if (img->trans.data == nullptr) {
            img->trans.data = allocator.allocate(scalar_bytes, alignof(float));
        }
        if (img->opacity.data == nullptr) {
            img->opacity.data = allocator.allocate(scalar_bytes, alignof(float));
        }
        if (img->depth.data == nullptr) {
            img->depth.data = allocator.allocate(scalar_bytes, alignof(float));
        }
    }
    if (mask_bytes > 0 && img->hitmask.data == nullptr) {
        img->hitmask.data = allocator.allocate(mask_bytes, alignof(uint32_t));
    }

    if ((image_bytes > 0 && img->image.data == nullptr) ||
        (scalar_bytes > 0 && (img->trans.data == nullptr || img->opacity.data == nullptr || img->depth.data == nullptr)) ||
        (mask_bytes > 0 && img->hitmask.data == nullptr)) {
        return HP_STATUS_OUT_OF_MEMORY;
    }

    buffers.image = static_cast<float*>(img->image.data);
    buffers.trans = static_cast<float*>(img->trans.data);
    buffers.opacity = static_cast<float*>(img->opacity.data);
    buffers.depth = static_cast<float*>(img->depth.data);
    buffers.hitmask = static_cast<uint32_t*>(img->hitmask.data);
    return HP_STATUS_SUCCESS;
}

bool tensor_is_device_vec3(const hp_tensor& tensor) {
    return tensor.memspace == HP_MEMSPACE_DEVICE &&
           tensor.dtype == HP_DTYPE_F32 &&
           tensor.rank >= 2 &&
           tensor.shape[tensor.rank - 1] == 3;
}

bool tensor_is_device_scalar(const hp_tensor& tensor) {
    return tensor.memspace == HP_MEMSPACE_DEVICE &&
           tensor.dtype == HP_DTYPE_F32 &&
           tensor.rank >= 1;
}

}  // namespace

hp_status img_generate_cuda(const hp_plan* plan,
                            const hp_intl_t* intl,
                            const hp_rays_t* rays,
                            hp_img_t* img,
                            void* ws,
                            size_t ws_bytes) {
    if (plan == nullptr || intl == nullptr || rays == nullptr || img == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    if (!tensor_is_device_vec3(intl->radiance) ||
        !tensor_is_device_scalar(intl->transmittance) ||
        !tensor_is_device_scalar(intl->opacity) ||
        !tensor_is_device_scalar(intl->depth) ||
        rays->pixel_ids.memspace != HP_MEMSPACE_DEVICE ||
        rays->pixel_ids.dtype != HP_DTYPE_U32) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    const size_t ray_count = (intl->transmittance.rank >= 1) ? static_cast<size_t>(intl->transmittance.shape[0]) : 0;
    const size_t pixel_count = static_cast<size_t>(plan->desc.width) * static_cast<size_t>(plan->desc.height);

    DeviceImgBuffers device_buffers{};
    const hp_status device_status = ensure_image_buffers_device(plan, img, ws, ws_bytes, device_buffers);
    if (device_status != HP_STATUS_SUCCESS) {
        return device_status;
    }

    std::vector<float> h_radiance(ray_count * 3U, 0.0f);
    std::vector<float> h_trans(ray_count, 0.0f);
    std::vector<float> h_opacity(ray_count, 0.0f);
    std::vector<float> h_depth(ray_count, 0.0f);
    std::vector<uint32_t> h_pixel_ids(ray_count, 0U);

    if (ray_count > 0) {
        if (cudaMemcpy(h_radiance.data(), intl->radiance.data, h_radiance.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(h_trans.data(), intl->transmittance.data, h_trans.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(h_opacity.data(), intl->opacity.data, h_opacity.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(h_depth.data(), intl->depth.data, h_depth.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(h_pixel_ids.data(), rays->pixel_ids.data, h_pixel_ids.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return HP_STATUS_INTERNAL_ERROR;
        }
    }

    hp_intl_t host_intl{};
    host_intl.radiance.data = h_radiance.data();
    host_intl.radiance.dtype = HP_DTYPE_F32;
    host_intl.radiance.memspace = HP_MEMSPACE_HOST;
    host_intl.radiance.rank = 2;
    host_intl.radiance.shape[0] = static_cast<int64_t>(ray_count);
    host_intl.radiance.shape[1] = 3;
    host_intl.radiance.stride[1] = 1;
    host_intl.radiance.stride[0] = 3;

    host_intl.transmittance.data = h_trans.data();
    host_intl.transmittance.dtype = HP_DTYPE_F32;
    host_intl.transmittance.memspace = HP_MEMSPACE_HOST;
    host_intl.transmittance.rank = 1;
    host_intl.transmittance.shape[0] = static_cast<int64_t>(ray_count);
    host_intl.transmittance.stride[0] = 1;

    host_intl.opacity = host_intl.transmittance;
    host_intl.opacity.data = h_opacity.data();
    host_intl.depth = host_intl.transmittance;
    host_intl.depth.data = h_depth.data();

    hp_rays_t host_rays{};
    host_rays.pixel_ids.data = h_pixel_ids.data();
    host_rays.pixel_ids.dtype = HP_DTYPE_U32;
    host_rays.pixel_ids.memspace = HP_MEMSPACE_HOST;
    host_rays.pixel_ids.rank = 1;
    host_rays.pixel_ids.shape[0] = static_cast<int64_t>(ray_count);
    host_rays.pixel_ids.stride[0] = 1;

    std::vector<std::byte> host_ws;
    const size_t image_bytes = pixel_count * 3U * sizeof(float);
    const size_t scalar_bytes = pixel_count * sizeof(float);
    const size_t mask_bytes = pixel_count * sizeof(uint32_t);
    const size_t host_ws_bytes = image_bytes + scalar_bytes * 3 + mask_bytes + 128;
    host_ws.resize(host_ws_bytes);

    hp_img_t host_img{};
    const hp_status cpu_status = img_generate_cpu(plan, &host_intl, &host_rays, &host_img, host_ws.data(), host_ws.size());
    if (cpu_status != HP_STATUS_SUCCESS) {
        return cpu_status;
    }

    if (pixel_count > 0) {
        if (cudaMemcpy(device_buffers.image, host_img.image.data, image_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(device_buffers.trans, host_img.trans.data, scalar_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(device_buffers.opacity, host_img.opacity.data, scalar_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(device_buffers.depth, host_img.depth.data, scalar_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(device_buffers.hitmask, host_img.hitmask.data, mask_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            return HP_STATUS_INTERNAL_ERROR;
        }
    }

    return HP_STATUS_SUCCESS;
}

}  // namespace hp_internal

#endif  // HP_WITH_CUDA
