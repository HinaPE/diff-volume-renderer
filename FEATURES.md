# dvren 功能清单

本文梳理了当前仓库（`diff-volume-renderer`）中已经落地的能力，涵盖 C++ 库 `libdvren`、关联的 CLI 程序和随附测试，可作为评估本项目可实现功能的参考。

## 核心库（libdvren）
- 上下文管理：`dvren::Context::Create` 封装了 `hp_ctx_create`，可选择设备与运行标志，并负责生命周期管理，确保与 Hotpath 引擎打通（`src/core/context.cpp:33`）。
- 渲染计划构建：`dvren::Plan::Create` 支持设置分辨率、近远裁剪、采样步长与最大步数、随机种子，并可选择固定/分层采样模式及 ROI、针孔/正交摄像机与自定义内参/外参（`src/core/plan.cpp:58`、`include/dvren/core/plan.hpp:18`）。
- 稠密体数据：`dvren::DenseGridField::Create` 可加载给定分辨率的密度与颜色体素数据，自定义 AABB、插值模式（线性/最近邻）和场外策略（置零/截断），并分配 Hotpath 兼容的场对象（`src/fields/dense_grid.cpp:69`）。
- 体素梯度处理：`DenseGridField::AccumulateSampleGradients` 将采样梯度按照当前插值与边界策略回流到体素网格，支持三线性权重（`src/fields/dense_grid.cpp:171`）。

## 渲染与结果
- 前向渲染管线：`dvren::Renderer::Forward` 依次调用 `hp_ray`→`hp_samp/hp_samp_int_fused`→`hp_int`→`hp_img`，支持 staged 与 fused 两种路径，自适应 ROI 尺寸并记录阶段耗时（`src/render/renderer.cpp:232`）。
- CUDA Graph 兼容：在启用 `RenderOptions::enable_graph` 且编译包含 CUDA 时，前向/反向自动尝试图捕获，不满足条件时会在统计信息中给出说明（`src/render/renderer.cpp:252`、`src/render/renderer.cpp:497`）。
- 前向输出：`ForwardResult` 返回 RGB 图像、透射率、遮挡率、深度、命中掩码，以及实际的光线/采样数量与阶段统计（`include/dvren/render/renderer.hpp:38`）。
- 工作区统计：`Renderer::workspace_info` 汇总光线、采样、积分、影像、梯度及临时缓冲的内存占用，便于容量规划（`src/render/renderer.cpp:572`）。

### 前向渲染示例
- CLI 最小示例：执行 `dvren_render examples/simple_volume.json build/out.ppm` 会加载 `examples/simple_volume.json` 中的 2×2×2 密度与颜色体素，使用固定步长 `dt=0.1`、最大 16 采样、启用 fused 路径生成 4×4 PPM 图像，同时在标准输出展示光线/采样计数与内存占用（`examples/simple_volume.json:1`、`apps/dvren_render/main.cpp:261`）。
- CLI 自定义参数：在 JSON `render` 节点中加入 `"roi": {"x": 0, "y": 0, "width": 256, "height": 128}`、`"sampling_mode": "stratified"`、`"options": {"use_fused_path": false, "enable_graph": true}` 可切换到分层采样 + staged 路径 + CUDA Graph 捕获模式；无需源码改动，解析逻辑会填充对应的 `PlanDescriptor` 与 `RenderOptions`（`apps/dvren_render/main.cpp:86`、`apps/dvren_render/main.cpp:92`、`apps/dvren_render/main.cpp:235`）。
- C++ 嵌入式调用：按照 README 示例创建 `dvren::Context`、`dvren::Plan`、`dvren::DenseGridField` 并构造 `dvren::Renderer renderer(ctx, plan, opts);` 后调用 `renderer.Forward(field, forward);` 可在应用内直接获取 RGB、深度等缓冲并读取 `forward.stats` 做性能分析（`README.md:177`、`src/render/renderer.cpp:232`）。

## 可微分能力
- 反向传播：`dvren::Renderer::Backward` 接收上游图像梯度，调用 `hp_diff` 生成采样梯度，并自动回写体素密度/颜色梯度和相机 3×4 Jacobian（`src/render/renderer.cpp:390`）。
- 梯度管理：在反向前会清零缓冲，并确保前向已经执行且采样数有效，避免非法调用；返回结果中包含累计的体素梯度与采样计数，方便外部优化器使用（`src/render/renderer.cpp:409`、`include/dvren/render/renderer.hpp:55`）。

## CLI 应用（`dvren_render`）
- JSON 配置：解析渲染与体数据配置，支持采样模式、ROI、摄像机模型/内外参/正交缩放、体素插值和 OOB 策略等字段，缺省时自动补齐（`apps/dvren_render/main.cpp:40`、`apps/dvren_render/main.cpp:111`）。
- 默认体颜色：当未提供颜色数组时，CLI 会自动将密度复制成灰度颜色，降低测试成本（`apps/dvren_render/main.cpp:146`）。
- 渲染输出：`RenderToFile` 创建渲染器、执行前向、将浮点图像写入 PPM（P6）文件，并输出光线/采样数量与各缓冲使用字节数（`apps/dvren_render/main.cpp:261`）。
- 命令行体验：`main` 入口支持 `dvren_render <config> [output.ppm]`，覆盖输出路径、打印错误信息并回显生成文件位置（`apps/dvren_render/main.cpp:314`）。

## 回归测试
- `tests/core/test_core.cpp` 运行 staged/fused/graph 三种模式，验证前向图像与梯度一致性、梯度正值累积以及工作区统计，保证核心差分渲染行为稳定（`tests/core/test_core.cpp:74`）。
