#include <torch/extension.h>
#include <tuple>

#ifdef WITH_CUDA
std::tuple<at::Tensor, at::Tensor> SampleVoge(
    const at::Tensor& image, // (N, W, H, C)
    const at::Tensor& vert_weight, // (N, W, H, K)
    const at::Tensor& vert_index,  // (N, W, H, K)
    const int num_vert
);

std::tuple<at::Tensor, at::Tensor> SampleVogeBackward(
    const at::Tensor& image, // (N, W, H, C)
    const at::Tensor& vert_weight, // (N, W, H, K)
    const at::Tensor& vert_index,  // (N, W, H, K)
    const at::Tensor& grad_feature, // (L, C)
    const at::Tensor& grad_weight_sum // (L, )
);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
