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
    const int B,
    const int C,
    const int H,
    const int W,
    const int K,
    float* vert_feature,  // (B, N, C)
    float* vert_weight_sum // (B, N)
){
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int pid = tid; pid < B * H * W * K; pid += num_threads) {
        const int bi = pid / (H * W * K);
        const int yi = (pid % (H * W * K)) / (W * K);
        const int xi = (pid % (W * K)) / K;

        const int idx_point = vert_index[pid] + bi * N;
        const float weight_ = vert_weight[pid];
        const int pixel_idx = bi * (W * H) + yi * W + xi;

        if (idx_point == -1){
            continue;
        }
        dotvectoratom(weight_, image + (pixel_idx * C), vert_feature + (idx_point * C), C);

        atomicAdd(vert_weight_sum + bi * N + idx_point, weight_);
    }
}


std::tuple<at::Tensor, at::Tensor> SampleVoge(
    const at::Tensor& image, // (B, W, H, C)
    const at::Tensor& vert_weight, // (B, W, H, K)
    const at::Tensor& vert_index,  // (B, W, H, K)
    const int num_vert
){
    at::cuda::CUDAGuard device_guard(image.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int B = image.size(0);
    const int H = image.size(1);
    const int W = image.size(2);
    const int C = image.size(3);
    const int K = vert_weight.size(3);
    const int N = num_vert;

    auto float_opts = vert_weight.options().dtype(at::kFloat);

    at::Tensor vert_feature = at::zeros({B, N, C}, float_opts);
    at::Tensor vert_weight_sum = at::zeros({B, N}, float_opts);

    const size_t blocks = 1024;
    const size_t threads = 64;

    SampleVogeKernel<<<blocks, threads, 0, stream>>>(
        image.contiguous().data_ptr<float>(),
        vert_weight.contiguous().data_ptr<float>(),
        vert_index.contiguous().data_ptr<int32_t>(),
        N,
        B,
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
    const int N,
    const int B,
    const int C,
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

    for (int pid = tid; pid < B * H * W * K; pid += num_threads) {
        const int bi = pid / (H * W * K);
        const int yi = (pid % (H * W * K)) / (W * K);
        const int xi = (pid % (W * K)) / K;

        const int idx_point = vert_index[pid] + bi * N;
        const float weight_ = vert_weight[pid];

        if (idx_point == -1){
            continue;
        }
        dotvectoratom(weight_, grad_feature + idx_point * C, grad_image + (bi * H * W + yi * W + xi) * C, C);
        
        float sum_grad = grad_weight_sum[idx_point];
        for(int c =0; c < C; ++c){
            sum_grad += grad_feature[idx_point * C + c] * image[(bi * H * W + yi * W + xi) * C + c];
        }
        atomicAdd(grad_vert_weight + pid, sum_grad);
    }
}


std::tuple<at::Tensor, at::Tensor> SampleVogeBackward(
    const at::Tensor& image, // (B, W, H, C)
    const at::Tensor& vert_weight, // (B, W, H, K)
    const at::Tensor& vert_index,  // (B, W, H, K)
    const at::Tensor& grad_feature, // (B, N, C)
    const at::Tensor& grad_weight_sum // (B, N, )
){
    at::cuda::CUDAGuard device_guard(image.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int B = image.size(0);
    const int H = image.size(1);
    const int W = image.size(2);
    const int C = image.size(3);
    const int K = vert_weight.size(3);
    const int N = grad_weight_sum.size(1);
    
    auto float_opts = vert_weight.options().dtype(at::kFloat);

    at::Tensor grad_image = at::zeros({B, H, W, C}, float_opts);
    at::Tensor grad_vert_weight = at::zeros({B, H, W, K}, float_opts);

    const size_t blocks = 1024;
    const size_t threads = 64;

    SampleVogeBackwardKernel<<<blocks, threads, 0, stream>>>(
        image.contiguous().data_ptr<float>(),
        vert_weight.contiguous().data_ptr<float>(),
        vert_index.contiguous().data_ptr<int32_t>(),
        N,
        B,
        C,
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