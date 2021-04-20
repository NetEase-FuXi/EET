#include "core/common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__
void softmax_kernel_gpt2(T *qk_buf_,const int64_t* __restrict padding_index, const int head_num, const int seq_len, const T scalar)
{
    int batch_id = blockIdx.x / head_num;
    int qk_offset = blockIdx.x * seq_len * seq_len;

    __shared__ float s_sum, s_max;

    for(int i = 0; i < seq_len; ++i)
    {
      float qk = threadIdx.x <= i ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
      int left_padding_len = 0;
      if (padding_index != nullptr){
          left_padding_len = padding_index[batch_id];
      }
      float tmp = (threadIdx.x <= i && threadIdx.x >= left_padding_len) ? (float)(qk * (float)scalar): -1e20f;

      float max_val = blockReduceMax<float>(tmp);

      if(threadIdx.x == 0)
        s_max = max_val;
      __syncthreads();

      qk = threadIdx.x <= i ? __expf(tmp - s_max) : 0.0f;

      float sum_val = blockReduceSum<float>(qk);

      if(threadIdx.x == 0)
      {
        s_sum = sum_val + 1e-6f;
      }
      __syncthreads();

      if(threadIdx.x < seq_len)
        qk_buf_[threadIdx.x + qk_offset] = threadIdx.x <= i ?  (T)(qk / s_sum) : (T)0.0f;

      qk_offset += seq_len;
    }
}


template <typename T>
__global__
void softmax_kernel_v2_gpt2(T *qk_buf_, const int64_t* __restrict padding_index, const int head_num, const int seq_len, const float scalar)
{
    int seq_id = blockIdx.x % seq_len;
    int qk_offset = blockIdx.x * seq_len;
    int batch_id = blockIdx.x / (seq_len * head_num);
    __shared__ float s_sum, s_max;

    float qk = threadIdx.x <= seq_id ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;

    int left_padding_len = 0;
    if (padding_index != nullptr){
        left_padding_len = padding_index[batch_id];
    }
    float tmp = (threadIdx.x <= seq_id && threadIdx.x >= left_padding_len) ? (float)(qk * (float)scalar) : -1e20f;
    float max_val = blockReduceMax<float>(tmp);
    if(threadIdx.x == 0)
      s_max = max_val;
    __syncthreads();

    float qk_tmp = threadIdx.x <= seq_id ? __expf((float)(tmp - s_max)) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);

    if(threadIdx.x == 0)
    {
      s_sum = sum_val + 1e-6f;
    }
    __syncthreads();

    if(threadIdx.x < seq_len)
      qk_buf_[threadIdx.x + qk_offset] = threadIdx.x <= seq_id ? (T)(qk_tmp / s_sum) : (T)0.0f;
}

template <typename T>
__global__
void softmax_kernel_v3_gpt2(T *qk_buf_, const int64_t* __restrict padding_index, const int head_num, const int seq_len, const T scalar)
{
    float tmp = -1e20f;
    int qk_offset;
    __shared__ float s_mean, s_max;

    int seq_id = blockIdx.x;
    int batch_id = blockIdx.y;

    qk_offset = ((blockIdx.y*head_num + blockIdx.z)*seq_len + blockIdx.x) *seq_len + threadIdx.x;
    if (threadIdx.x <= seq_id){
        float qk = static_cast<float>(qk_buf_[qk_offset]);
        tmp = qk * static_cast<float>(scalar);
    }

    float max_val = blockReduceMax<float>(tmp);
    if (threadIdx.x == 0){
        s_max = max_val;
    }
    __syncthreads();
    
    int left_padding_len = 0;
    if (padding_index != nullptr){
        left_padding_len = padding_index[batch_id];
    }

    float qk_tmp = (threadIdx.x <= seq_id && threadIdx.x >= left_padding_len) ? __expf(tmp - s_max) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);
    if (threadIdx.x == 0){
        s_mean = sum_val + 1e-6f;
        s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();

    if(threadIdx.x < seq_len)
      qk_buf_[qk_offset] = threadIdx.x <= seq_id ? (T)(qk_tmp * s_mean) : (T)0.0f;
}

template <typename T>
__global__
void
softmax_kernel_v3_LE32_gpt2(T *qk_buf_, const int64_t* __restrict padding_index, const int head_num, const int seq_len, const T scalar)
{

    int qk_offset;
    __shared__ float s_mean, s_max;
    float tmp = -1e20f;

    int seq_id = blockIdx.x;
    int batch_id = blockIdx.y;

    qk_offset = ((blockIdx.y*head_num + blockIdx.z)*seq_len + blockIdx.x) *seq_len + threadIdx.x;
    if (threadIdx.x <= seq_id){
        float qk = static_cast<float>(qk_buf_[qk_offset]);
        tmp = static_cast<float>(qk) * static_cast<float>(scalar);
    }
    float max_val = warpReduceMax<float>(tmp);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    int left_padding_len = 0;
    if (padding_index != nullptr){
        left_padding_len = padding_index[batch_id];
    }
    tmp = (threadIdx.x <= seq_id && threadIdx.x >= left_padding_len) ? __expf(tmp - s_max) : 0.0f;
    float sum_val = warpReduceSum<float>(tmp);

    if (threadIdx.x == 0){
      s_mean = sum_val + 1e-6f;
      s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();

    if(threadIdx.x < seq_len)
      qk_buf_[qk_offset] = threadIdx.x <= seq_id ? (T)(tmp * s_mean) : (T)0.0f;
}

template <class T>
void softmax_kernel(void *qk_buf_, const int64_t* __restrict padding_index, const int& batch_size, 
                    const int& head_num, const int& seq_len, const float& scalar, const cudaStream_t stream){
  dim3 grid, block;

  //deal with odd seq_len
  if (seq_len % 2 != 0){
    if(seq_len <= 32)
      block.x = 32;
    else if(seq_len > 32 && seq_len <= 64)
      block.x = 64;
    else if(seq_len > 64 && seq_len <= 128)
      block.x = 128;
    else if(seq_len > 128 && seq_len <= 256)
      block.x = 256;
    else if(seq_len > 256 && seq_len <= 512)
      block.x = 512;
    else
      block.x = 1024;

    if(batch_size * head_num <= 120)
    {
      grid.x = batch_size * head_num * seq_len;
      softmax_kernel_v2_gpt2 <T><<<grid, block, 0, stream>>>((T*)qk_buf_, padding_index, head_num, seq_len, scalar);
    }
    else
    {
      grid.x = batch_size * head_num;
      softmax_kernel_gpt2 <T><<<grid, block, 0, stream>>>((T*)qk_buf_, padding_index, head_num, seq_len, scalar);
    }
  }
  //deal with even seq_len 
  else{
    grid.x = seq_len;
    grid.y = batch_size;
    grid.z = head_num;
    if (seq_len <= 32){
      block.x = 32;
      softmax_kernel_v3_LE32_gpt2 <T><<<grid, block, 0, stream>>>((T*)qk_buf_, padding_index, head_num, seq_len, scalar);
    }
    else{
      block.x = (seq_len + 31) / 32 * 32;
      softmax_kernel_v3_gpt2 <T><<<grid, block, 0, stream>>>((T*)qk_buf_, padding_index, head_num, seq_len, scalar);
      }
    }
  }

template void softmax_kernel<float>(void *qk_buf_, const int64_t* padding_index,const int& batch_size, 
                                      const int& head_num, const int& seq_len, const float& scalar, const cudaStream_t stream);
template void softmax_kernel<half>(void *qk_buf_, const int64_t* padding_index, const int& batch_size, 
                                      const int& head_num, const int& seq_len, const float& scalar, const cudaStream_t stream);
