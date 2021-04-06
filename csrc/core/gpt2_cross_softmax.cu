#include "core/common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

// cross softmax code modified from Nvidia's DeepLearningExamples
// https://github.com/NVIDIA/DeepLearningExamples/blob/master/FasterTransformer/v3.1/fastertransformer/cuda/open_attention.cu#L1399-L1583


template <typename T>
__global__
void cross_softmax_kernel_opt(T *qk_buf_, const int head_num, const int seq_len,const int mem_seq_len, const T scalar)
{
    int qk_offset = blockIdx.x * seq_len * mem_seq_len;

    __shared__ float s_sum, s_max;

    for(int i = 0; i < mem_seq_len; ++i)
    {
        float qk = threadIdx.x < mem_seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;

        float tmp = threadIdx.x < mem_seq_len ? (float)(qk * (float)scalar): -1e20f;

        float max_val = blockReduceMax<float>(tmp);

        if(threadIdx.x == 0)
            s_max = max_val;
        __syncthreads();

        qk = threadIdx.x < mem_seq_len ? __expf(tmp - s_max) : 0.0f;

        float sum_val = blockReduceSum<float>(qk);

        if(threadIdx.x == 0)
        {
            s_sum = sum_val + 1e-6f;
        }
        __syncthreads();

        if(threadIdx.x < mem_seq_len)
            qk_buf_[threadIdx.x + qk_offset] = (T)(qk / s_sum);

        qk_offset += mem_seq_len;
    }
}

template <typename T>
__global__
void cross_softmax_kernel_v2(T *qk_buf_, const int head_num, const int mem_seq_len, const float scalar)
{
    int qk_offset = blockIdx.x * mem_seq_len;

    __shared__ float s_sum, s_max;

    float qk = threadIdx.x < mem_seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;

    float tmp = threadIdx.x < mem_seq_len ? (float)(qk * (float)scalar) : -1e20f;
    float max_val = blockReduceMax<float>(tmp);
    if(threadIdx.x == 0)
        s_max = max_val;
    __syncthreads();

    float qk_tmp = threadIdx.x < mem_seq_len ? __expf((float)(tmp - s_max)) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);

    if(threadIdx.x == 0)
    {
        s_sum = sum_val + 1e-6f;
    }
    __syncthreads();

    if(threadIdx.x < mem_seq_len)
        qk_buf_[threadIdx.x + qk_offset] = (T)(qk_tmp / s_sum);
}

template <typename T>
__global__
void cross_softmax_kernel_v3(T *qk_buf_, const int head_num, const int seq_len,const int mem_seq_len, const T scalar)
{

    float tmp = -1e20f;
    int qk_offset;
    __shared__ float s_mean, s_max;
    if (threadIdx.x < mem_seq_len){
        qk_offset = ((blockIdx.y*head_num + blockIdx.z)*seq_len + blockIdx.x) *mem_seq_len + threadIdx.x;

        float qk = static_cast<float>(qk_buf_[qk_offset]);


        tmp = qk * static_cast<float>(scalar);
    }

    float max_val = blockReduceMax<float>(tmp);
    if (threadIdx.x == 0){
        s_max = max_val;
    }
    __syncthreads();

    float qk_tmp = threadIdx.x < mem_seq_len ? __expf(tmp - s_max) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);
    if (threadIdx.x == 0){
        s_mean = sum_val + 1e-6f;
        s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();

    if(threadIdx.x < mem_seq_len)
        qk_buf_[qk_offset] = (T)(qk_tmp * s_mean);
}

template <>
__global__
void cross_softmax_kernel_v3(half *qk_buf_, const int head_num, const int seq_len,const int mem_seq_len, const half scalar)
{
    half2* qk_buf_half2Ptr = (half2*) qk_buf_;

    int qk_offset;
    int threadIdx2 = threadIdx.x << 1;
    __shared__ float s_mean, s_max;
    half2 tmp = __float2half2_rn(0.0f);

    float max_val = -1e20f;
    half2 qk;
    if (threadIdx2 < mem_seq_len){
        qk_offset = ((((blockIdx.y*head_num + blockIdx.z)*seq_len + blockIdx.x) *mem_seq_len) >> 1) + threadIdx.x;

        qk = qk_buf_half2Ptr[qk_offset];
        tmp = __hmul2(__half2half2(scalar), qk);
        max_val = fmax((float)tmp.x, (float)tmp.y);
    }

    max_val = blockDim.x <= 32 ? warpReduceMax(max_val) : blockReduceMax<float>(max_val);

    if (threadIdx.x == 0){
        s_max = max_val;
    }
    __syncthreads();

    if (threadIdx2 < mem_seq_len){
        tmp = h2exp(__hsub2(tmp, __float2half2_rn(s_max)));
    }
    float sum_val = blockDim.x <= 32 ? warpReduceSum((float)(tmp.x + tmp.y)) : blockReduceSum<float>((float)(tmp.x + tmp.y));

    if (threadIdx.x == 0){
        s_mean = sum_val + 1e-6f;
        s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();

    if(threadIdx2 < mem_seq_len){
        qk = __hmul2(tmp, __float2half2_rn(s_mean));
        qk_buf_half2Ptr[qk_offset] = qk;
    }
}

template <typename T>
__global__
void cross_softmax_kernel_v3_LE32(T *qk_buf_, const int head_num, const int seq_len, const int mem_seq_len,const T scalar)
{

    int qk_offset;
    __shared__ float s_mean, s_max;
    float tmp = -1e20f;
    if (threadIdx.x < mem_seq_len){
        qk_offset = ((blockIdx.y*head_num + blockIdx.z)*seq_len + blockIdx.x) *mem_seq_len + threadIdx.x;

        float qk = static_cast<float>(qk_buf_[qk_offset]);


        tmp = static_cast<float>(qk) * static_cast<float>(scalar);
    }
    float max_val = warpReduceMax<float>(tmp);

    if (threadIdx.x == 0){
        s_max = max_val;
    }
    __syncthreads();

    tmp = threadIdx.x < mem_seq_len ? __expf(tmp - s_max) : 0.0f;
    float sum_val = warpReduceSum<float>(tmp);

    if (threadIdx.x == 0){
        s_mean = sum_val + 1e-6f;
        s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();

    if(threadIdx.x < mem_seq_len)
        qk_buf_[qk_offset] = (T)(tmp * s_mean);
}

template <class T>
void cross_softmax_kernel(void *qk_buf_, const int &batch_size,
                    const int &head_num, const int &seq_len, const int& mem_seq_len,const float &scalar, const cudaStream_t stream)
{
  dim3 grid, block;

  //deal with odd seq_len
  if (seq_len % 2 != 0)
  {
    if (seq_len <= 32)
      block.x = 32;
    else if (seq_len > 32 && seq_len <= 64)
      block.x = 64;
    else if (seq_len > 64 && seq_len <= 128)
      block.x = 128;
    else if (seq_len > 128 && seq_len <= 256)
      block.x = 256;
    else if (seq_len > 256 && seq_len <= 512)
      block.x = 512;
    else
      block.x = 1024;

    if (batch_size * head_num <= 120)
    {
      grid.x = batch_size * head_num * seq_len;
      cross_softmax_kernel_v2<T><<<grid, block, 0, stream>>>((T*)qk_buf_, head_num, mem_seq_len, scalar);
    }
    else
    {
      grid.x = batch_size * head_num;
      cross_softmax_kernel_opt<T><<<grid, block, 0, stream>>>((T*)qk_buf_, head_num, seq_len,mem_seq_len, scalar);
    }
  }
  //deal with even seq_len
  else
  {
    grid.x = seq_len;
    grid.y = batch_size;
    grid.z = head_num;
    if (seq_len <= 32)
    {
      block.x = 32;
      cross_softmax_kernel_v3_LE32<T><<<grid, block, 0, stream>>>((T*)qk_buf_, head_num, seq_len,mem_seq_len, scalar);
    }
    else
    {
      if (sizeof(T) == sizeof(float))
      {
        block.x = (seq_len + 31) / 32 * 32;
        cross_softmax_kernel_v3<T><<<grid, block, 0, stream>>>((T*)qk_buf_, head_num, seq_len,mem_seq_len, scalar);
      }
      else
      {
        block.x = (seq_len / 2 + 31) / 32 * 32;
        cross_softmax_kernel_v3<T><<<grid, block, 0, stream>>>((T*)qk_buf_, head_num, seq_len,mem_seq_len, scalar);
      }
    }
  }
}

template void cross_softmax_kernel<float>(void *qk_buf_,const int& batch_size, 
                                      const int& head_num, const int& seq_len,const int& mem_seq_len, const float& scalar, const cudaStream_t stream);

template void cross_softmax_kernel<half>(void *qk_buf_, const int& batch_size, 
                                      const int& head_num, const int& seq_len,const int& mem_seq_len, const float& scalar, const cudaStream_t stream);

