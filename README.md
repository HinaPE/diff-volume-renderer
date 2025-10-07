# dvren — Simple-in Simple-out Differentiable Volume Renderer

[![Windows](https://github.com/HinaPE/diff-volume-renderer/actions/workflows/windows-build.yml/badge.svg)](https://github.com/HinaPE/diff-volume-renderer/actions/workflows/windows-build.yml)
[![Linux](https://github.com/HinaPE/diff-volume-renderer/actions/workflows/linux-build.yml/badge.svg)](https://github.com/HinaPE/diff-volume-renderer/actions/workflows/linux-build.yml)

A high-performance CUDA volume renderer designed to be a small black box: give inputs in one standard form, get outputs in one standard form. It renders explicit volumes (dense grid, NanoVDB planned) and implicit neural fields (tiny-cuda-nn) under the same API, and supports reverse-mode differentiation for optimization and physics-informed pipelines.

---

## 0. Phase P0 快速演示

P0 已新增主工程封装与命令行渲染器。构建成功后可直接运行：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target dvren_render
./build/Release/dvren_render.exe examples/simple_volume.json output.ppm
```

`examples/simple_volume.json` 提供了一个 2×2×2 的最小体积示例，生成的 `output.ppm` 为 PPM(P6) 格式。命令行工具完全依赖 CPU 路径，可在没有 GPU 的环境下验证主流程。

---

## 1. 设计目标

* Simple in, simple out: 单一最小 API，显式与隐式场共享同一渲染/求导接口。
* 最高性能优先：纯 CUDA 实现，面向大批量射线与体数据，支持早停、后续将加入空洞跳过与稀疏数据结构。
* 可微：前向体积分 + 反向链路，提供对体素密度与颜色参数的梯度；隐式网络场将打通到其参数梯度。
* 可插拔场源：Grid_Dense、NanoVDB(计划)、tiny-cuda-nn(计划) 使用同一 FieldProvider。
* 工程可控：C++23 + CUDA，尽量少依赖；OpenVDB 与 NanoVDB 走一个开关；torch 绑定为可选项。

---

## 2. 目标输入/输出

### 2.1 输入

* `RaysSOA`

    * `width, height`
    * `origins`  float[W*H*3]
    * `dirs`     float[W*H*3] (单位方向, 世界系)
    * `tmin`     float[W*H]
    * `tmax`     float[W*H]
* `SamplingSpec`

    * `n_steps`  固定步长采样次数
    * `dt`       采样步长
    * `sigma_scale`  密度缩放
    * `stop_thresh`  早停阈值(1 - T)
* `FieldProvider`

    * `type`     场类型 (目前: `Field_Grid_Dense`; 计划: `Field_NanoVDB`, `Field_NN_TCNN`)
    * `device_ctx`  设备端上下文指针(各适配器自管理)
* `RenderFlags`

    * `retain_rgb`       预留
    * `use_fp16_field`   预留/未来优化

### 2.2 输出

* `ForwardOutputs`

    * `image` float[W*H*3] 行主序 RGB
    * `saved_ctx`        设备端保存的最小上下文(反向用)
    * `saved_ctx_bytes`  上下文字节数

### 2.3 反向的梯度输出

* 对显式体素:

    * `g_sigma`  float[Nx*Ny*Nz]
    * `g_rgb`    float[Nx*Ny*Nz*3]
    * 获取方式: 先 `field_grid_dense_zero_grad()`，调用 `volume_forward()` 与 `volume_backward()`，最后 `field_grid_dense_download_grad()` 拉回主机。
* 对隐式网络(计划):

    * 将透传每步样本的需求到 tiny-cuda-nn，或采用重算策略并调用网络的 backward，实现参数梯度回传。

---

## 3. 最小 API

```cpp
namespace dvren {

struct RaysSOA { int width, height; const float* origins; const float* dirs; const float* tmin; const float* tmax; };
struct SamplingSpec { int n_steps; float dt; float sigma_scale; float stop_thresh; };
struct ForwardOutputs { std::vector<float> image; void* saved_ctx; size_t saved_ctx_bytes; };
enum FieldType : uint32_t { Field_Grid_Dense = 1 };
struct FieldProvider { FieldType type; void* device_ctx; };
struct RenderFlags { uint32_t retain_rgb; uint32_t use_fp16_field; };

bool volume_forward(const RaysSOA&, const SamplingSpec&, const FieldProvider&, const RenderFlags&, ForwardOutputs&);
bool volume_backward(void* saved_ctx, size_t saved_ctx_bytes, const float* dL_dimage, int width, int height, const FieldProvider&);

struct GridDenseDesc {
  int nx, ny, nz;
  float bbox_min[3];
  float bbox_max[3];
  const float* host_sigma;
  const float* host_rgb;
};

bool field_grid_dense_create(const GridDenseDesc&, FieldProvider& out);
void field_grid_dense_destroy(FieldProvider&);
void field_grid_dense_zero_grad(FieldProvider&);
bool field_grid_dense_download_grad(const FieldProvider&, std::vector<float>& sigma_g, std::vector<float>& rgb_g);

}
```

---

## 4. 典型调用流程

### 4.1 显式体渲染与求导

```cpp
dvren::FieldProvider fp;
dvren::GridDenseDesc gd;
// 设置 gd.nx,ny,nz; gd.bbox_min/max; gd.host_sigma/rgb
field_grid_dense_create(gd, fp);

dvren::RaysSOA rays{W,H,origins,dirs,tmin,tmax};
dvren::SamplingSpec spec{N, dt, sigma_scale, stop_thresh};
dvren::RenderFlags flags{0,1};
dvren::ForwardOutputs out;

field_grid_dense_zero_grad(fp);
volume_forward(rays, spec, fp, flags, out);

// dL/dimage 在主机侧计算或来自上游
volume_backward(out.saved_ctx, out.saved_ctx_bytes, dL_dimage, W, H, fp);

std::vector<float> gsig, grgb;
field_grid_dense_download_grad(fp, gsig, grgb);

field_grid_dense_destroy(fp);
```

### 4.2 从 .vdb 读入再渲染

提供辅助函数将 OpenVDB 采样为致密网格并复用同一渲染/反向通道。

```cpp
dvren::VdbLoadParams p;
// 设置 p.path, p.density_grid, p.nx,ny,nz, has_color 等
dvren::VdbDense dense;
vdb_load_to_dense(p, dense);

dvren::FieldProvider fp;
field_grid_dense_create(dense.desc, fp);

// 后续与 4.1 相同
```

CLI 示例:

```
dvren <vdb_path> <density_grid_name>
```

---

## 5. 内部逻辑架构

```
dataset/HostPack   ->  用户/数据集模块(已完成, 外部库)
                     (输出相机与数据流, 由调用方转换为 RaysSOA)

RaysSOA + SamplingSpec + FieldProvider
         |
         v
   Core Renderer (CUDA)
   - forward: 固定步长 + 早停; T,C 使用 double 累加; 三线性采样
   - backward: 重算采样; 反向体积分; 对体素参数做原子加散射
         |
         v
  ForwardOutputs(image, saved_ctx)
         |
         +-- backward(dL_dimage, saved_ctx) -> gradients in field adapter

Field Providers
- Grid_Dense: 设备端结构体 + 密度/颜色与梯度缓冲
- NanoVDB (计划): 设备只读稀疏体; 直接采样稀疏树, 避免致密展开
- NN_TCNN (计划): tiny-cuda-nn 前向/反向适配器; 支持 hashgrid/NGP MLP
```

关键点:

* SavedCtx 存储渲染所需的最小设备端指针与参数(分辨率、步数、dt、场上下文、设备端 rays)。生命周期由调用方管理。
* 反向使用重算策略，避免保存全量样本缓存; 对显式体素用原子加写入 `g_sigma`, `g_rgb`。
* 统一世界坐标: 射线与体素 bbox 在同一世界系；Dense grid 内部做归一化采样。

---

## 6. 数值与正确性

* 提供 `tests/test_grad` 用有限差分对单体素做数值校验。
* 建议在验证时关闭早停、增大 n_steps 并缩小 dt，选择射线必经体素，使用较大的 eps，以避免 float 量化吞噬信号。
* 核内 T 与颜色累积采用 double 中间值，输出图像为 float。

---

## 7. 性能策略

已实现:

* CUDA 核心体积分，T 与颜色 double 累加减少舍入误差。
* 早停 `stop_thresh`。

计划:

* 空体素跳过: occupancy bitfield 或 mip occupancy，加速空域。
* 稀疏数据结构: NanoVDB 直接设备采样，避免 dense 展开内存爆炸。
* 自适应步长 dt 或 cone tracing。
* CUDA Graph、持久化线程等进一步优化。
* tcnn 隐式场前向/反向融合，避免 host 往返。

---

## 8. 构建与依赖

* 语言/工具链: C++23, CUDA 11+ (建议算力 7.5/8.0/8.6/8.9/9.0)
* 可选依赖:

    * OpenVDB + NanoVDB (`DVREN_WITH_VDB=ON`) 用于从 .vdb 读取与工具链支持
    * tiny-cuda-nn (`DVREN_WITH_TCNN=ON`) 用于隐式神经场
    * TBB, zlib, Boost (OpenVDB 需要)
* 典型 CMake 选项:

    * `DVREN_WITH_VDB` ON/OFF
    * `DVREN_WITH_TCNN` ON/OFF
    * `DVREN_FAST_MATH`, `DVREN_USE_GRAPH`, `DVREN_FP16_FIELD` 等

---

## 9. 坐标与数据约定

* Grid_Dense:

    * 体素中心采样; 三线性插值
    * `bbox_min`, `bbox_max` 为世界坐标
    * `sigma` 为每体素的密度; `rgb` 为发射颜色
* RaysSOA:

    * `origins`, `dirs` 世界系; `dirs` 归一化
    * `tmin`, `tmax` 定义采样区间; 通常与体 bbox 交集

---

## 10. 与上游/下游集成

* 上游数据集模块: 将 HostPack 或外部数据转换为 RaysSOA 与 FieldProvider
* 下游优化器/损失:

    * 将 `image` 送入损失计算
    * 形成 `dL/dimage` 并调用 `volume_backward`
    * 显式体素使用下载的 `g_sigma`, `g_rgb`; 隐式网络通过其适配器拿到参数梯度

---

## 11. 路线图

* NanoVDB 设备采样适配器 `Field_NanoVDB`
* tiny-cuda-nn 隐式场适配器 `Field_NN_TCNN` 前向/反向
* Occupancy 跳过与层级空域加速
* Torch/Python 绑定 (`DVREN_BUILD_TORCH`) 用作 Autograd Op
* 可选物理损失 hook: 在反向前后注入 PDE 残差或约束项

---

## 12. 已知限制

* 当前反向仅对显式 Dense grid 输出体素梯度
* 仅支持单次散射的发射吸收模型
* 未实现运动模糊与时间维
* 双向渲染与多相函数未覆盖

---

## 13. 许可证与致谢

* 许可证暂定占位
* 致谢: OpenVDB, NanoVDB, tiny-cuda-nn, 以及社区渲染与可微图形相关工作

---

## 14. 快速问题排查

* 数值梯度全 0: eps 太小或该体素对选定像素影响接近 0，请选择穿越体素的像素并增大 eps。
* OpenVDB 构建困难: 推荐使用预编译 Boost 或包管理器；或转为仅 Dense grid 测试路径。
* 反向报错无梯度缓冲: 先调用 `field_grid_dense_zero_grad()` 再做 forward/backward。

---

如需我先实现 NanoVDB 设备采样适配器，或先做 occupancy 跳过以提升速度，请直接说明你的优先级。
