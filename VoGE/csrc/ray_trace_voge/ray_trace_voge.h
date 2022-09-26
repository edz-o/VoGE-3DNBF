
#pragma once

#include <torch/extension.h>
#include <tuple>

#ifdef WITH_CUDA
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> RayTraceFineVoge(
    const at::Tensor& mus,  // (P, 3)
    const at::Tensor& isigmas, // (P, 3, 3)
    const at::Tensor& rays, // (N, H, W, 3)
    const at::Tensor& bin_points, // (N, BH, BW, T)
    const float thr_act, // -log(thr + 1e-10)
    const int bin_size,
    const int K
);

std::tuple<at::Tensor, at::Tensor, at::Tensor> RayTraceFineVogeBackward(
    const at::Tensor& mus,  // (P, 3)
    const at::Tensor& isigmas, // (P, 3, 3)
    const at::Tensor& rays, // (N, H, W, 3)
    const at::Tensor& point_idxs, // (N, H, W, K)
    const at::Tensor& grad_len,  // (N, H, W, K)
    const at::Tensor& grad_act, // (N, H, W, K)
    const at::Tensor& grad_dsd // (N, H, W, K)
);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
