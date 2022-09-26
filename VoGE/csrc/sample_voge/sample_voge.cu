#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda.h>
#include <sstream>
#include <tuple>


__device__ void inline dotvectoratom(
    const float val,
    const float* source,
    float* target,
    int32_t size
){
    for (int i = 0; i < size; ++i){
        atomicAdd(target + i, source[i] * val);
    }
}

__global__ void SampleVogeKernel(
    const float* image, 
    const float* vert_weight,
    const int32_t* vert_index, 
    const int N,
    const int C,
    const int H,
    const int W,
    const int K,
    float* vert_feature,  // (L, C)
    float* vert_weight_sum // (L)
){
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int pid = tid; pid < H * W * K; pid += num_threads) {
        const int n = pid / (H * W * K);
        const int yi = (pid % (H * W * K)) / (W * K);
        const int xi = (pid % (W * K)) / K;

        const int idx_point = vert_index[pid];
        const float weight_ = vert_weight[pid];
        const int pixel_idx = n * H * W + yi * W + xi;

        if (idx_point == -1){
            continue;
        }
        dotvectoratom(weight_, image + (pixel_idx * C), vert_feature + (idx_point * C), C);

        atomicAdd(vert_weight_sum + idx_point, weight_);
    }
}


std::tuple<at::Tensor, at::Tensor> SampleVoge(
    const at::Tensor& image, // (N, W, H, C)
    const at::Tensor& vert_weight, // (N, W, H, K)
    const at::Tensor& vert_index,  // (N, W, H, K)
    const int num_vert
){
    at::cuda::CUDAGuard device_guard(image.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int N = image.size(0);
    const int H = image.size(1);
    const int W = image.size(2);
    const int C = image.size(3);
    const int K = vert_weight.size(3);
    const int L = num_vert;

    auto float_opts = vert_weight.options().dtype(at::kFloat);

    at::Tensor vert_feature = at::zeros({L, C}, float_opts);
    at::Tensor vert_weight_sum = at::zeros({L}, float_opts);

    const size_t blocks = 1024;
    const size_t threads = 64;

    SampleVogeKernel<<<blocks, threads, 0, stream>>>(
        image.contiguous().data_ptr<float>(),
        vert_weight.contiguous().data_ptr<float>(),
        vert_index.contiguous().data_ptr<int32_t>(),
        N,
        C,
        H,
        W,
        K,
        vert_feature.contiguous().data_ptr<float>(),
        vert_weight_sum.contiguous().data_ptr<float>()
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(vert_feature, vert_weight_sum);
}


__global__ void SampleVogeBackwardKernel(
    const float* image, 
    const float* vert_weight,
    const int32_t* vert_index, 
    const int C,
    const int N,
    const int H,
    const int W,
    const int K,
    const float* grad_feature, 
    const float* grad_weight_sum, 
    float* grad_image,
    float* grad_vert_weight
){
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int pid = tid; pid < H * W * K; pid += num_threads) {
        const int n = pid / (H * W * K);
        const int yi = (pid % (H * W * K)) / (W * K);
        const int xi = (pid % (W * K)) / K;
        const int k = pid % K;

        const int idx_point = vert_index[pid];
        const float weight_ = vert_weight[pid];
        const int pixel_idx = n * H * W + yi * W + xi;

        if (idx_point == -1){
            continue;
        }
        dotvectoratom(weight_, grad_feature + idx_point * C, grad_image + pixel_idx * C, C);
        
        float sum_grad = grad_weight_sum[idx_point];
        for(int c =0; c < C; ++c){
            sum_grad += grad_feature[idx_point * C + c] * image[pixel_idx * C + c];
        }
        atomicAdd(grad_vert_weight + pid, sum_grad);
    }
}


std::tuple<at::Tensor, at::Tensor> SampleVogeBackward(
    const at::Tensor& image, // (N, W, H, C)
    const at::Tensor& vert_weight, // (N, W, H, K)
    const at::Tensor& vert_index,  // (N, W, H, K)
    const at::Tensor& grad_feature, // (L, C)
    const at::Tensor& grad_weight_sum // (L, )
){
    at::cuda::CUDAGuard device_guard(image.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int N = image.size(0);
    const int H = image.size(1);
    const int W = image.size(2);
    const int C = image.size(3);
    const int K = vert_weight.size(3);
    
    auto float_opts = vert_weight.options().dtype(at::kFloat);

    at::Tensor grad_image = at::zeros({N, H, W, C}, float_opts);
    at::Tensor grad_vert_weight = at::zeros({N, H, W, K}, float_opts);

    const size_t blocks = 1024;
    const size_t threads = 64;

    SampleVogeBackwardKernel<<<blocks, threads, 0, stream>>>(
        image.contiguous().data_ptr<float>(),
        vert_weight.contiguous().data_ptr<float>(),
        vert_index.contiguous().data_ptr<int32_t>(),
        C,
        N,
        H,
        W,
        K,
        grad_feature.contiguous().data_ptr<float>(),
        grad_weight_sum.contiguous().data_ptr<float>(),
        grad_image.data_ptr<float>(),
        grad_vert_weight.data_ptr<float>()
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(grad_image, grad_vert_weight);
}
