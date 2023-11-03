#include "core/common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 32  // Define the block size

constexpr int BLOCK_SIZE_512 = 512;
constexpr int BLOCK_SIZE_256 = 256;
constexpr int BLOCK_SIZE_128 = 128;


__global__ void int8ToFloat(const int8_t*  A, half* B, const float* scales, int width, int height) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < height && col < width) {
        float val = (float)A[row * width + col] * scales[col];
        B[row * width + col] = __float2half(val);
    }
}


__global__ void dequantizeKernel(int8_t *input, float *scales, half *output, int width, int height) {
    __shared__ float shared_scales[BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    if (ty == 0 && col < width) {
        shared_scales[tx] = scales[col];
    }

    __syncthreads();

    if (row < height && col < width) {
        output[row * width + col] = __float2half((float)input[row * width + col] * shared_scales[tx]);
    }
}

template <int block_size>
__global__ void dequantizeKernel_v2(const int8_t *input, half *output, const float *scales, int width, int height) {
    int idx = blockIdx.x * width;
    int tid = threadIdx.x;
    #pragma unroll
    for (int col_id = tid; col_id < width; col_id += block_size) {
        output[idx + col_id] = __float2half((float)input[idx + col_id] * scales[col_id]);
    }
}

template <int block_size, int pack_size>
__global__ void dequantizeKernel_v3(int8_t *input, half *output, float *scales, int width, int height) {
    int idx = blockIdx.x * width / pack_size;
    int tid = threadIdx.x;
    assert(width % pack_size == 0);
    const int num_packs = width / pack_size;
    Vec_n_t<int8_t, pack_size> src_pack;
    Vec_n_t<float, pack_size> scale_pack;
    Vec_n_t<half, pack_size> dst_pack;
    VecType<int8_t, pack_size> *src_ptr = reinterpret_cast<VecType<int8_t, pack_size> *>(input);
    VecType<float, pack_size> *scale_ptr = reinterpret_cast<VecType<float, pack_size> *>(scales);
    VecType<half, pack_size> *dst_ptr = reinterpret_cast<VecType<half, pack_size> *>(output);

#pragma unroll
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
        src_pack.storage = src_ptr[idx + pack_id];
        scale_pack.storage = scale_ptr[pack_id];
#pragma unroll
        for (int i = 0; i < pack_size; ++i) {
            dst_pack.elem[i] = __float2half((float)src_pack.elem[i] * scale_pack.elem[i]);
        }
        dst_ptr[idx + pack_id] = dst_pack.storage;
    }
}

void dequant_weight(void *weight,void *dqweight, void *scale, int width, int height, const cudaStream_t stream)
{
    // dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    // dequantizeKernel<<<dimGrid, dimBlock,0,stream>>>((int8_t*)weight, (float*)scale, (half*)dqweight, width, height);


    // dim3 dimBlock(32, 32);
    // dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    // int8ToFloat<<<dimGrid, dimBlock,0,stream>>>((int8_t*)weight, (half*)dqweight, (float*)scale, width, height);
    
    dim3 dimBlock(BLOCK_SIZE_512);
    dim3 dimGrid(height);
    dequantizeKernel_v3<BLOCK_SIZE_512, 4><<<dimGrid, dimBlock, 0, stream>>>((int8_t*)weight, (half*)dqweight, (float*)scale, width, height);

}


void dequant_weight(void *weight,void *dqweight,void *scale,int width, int height, const cudaStream_t stream);
