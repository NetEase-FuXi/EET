#include "core/common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__
void softmax_kernel_gpt2(T *qk_buf,const int64_t* __restrict padding_index, const int head_num, const int seq_len, const T scalar)
{
    int batch_id = blockIdx.x / head_num;
    int qk_offset = blockIdx.x * seq_len * seq_len;

    __shared__ float s_sum, s_max;
    
    int left_padding_len = 0;
    if (padding_index != nullptr){
        left_padding_len = padding_index[batch_id];
    }

    // This is added here because if it is not added, the original value will be used by default, and the value will get bigger and bigger when there are many layers until it overflows during the calculation
    for (int i = 0; i < left_padding_len; i++)
    {
      if(threadIdx.x < seq_len)
        qk_buf[threadIdx.x + qk_offset] = (T)0.0f;
      qk_offset += seq_len;
    }
    
    for(int i = left_padding_len; i < seq_len; ++i)
    {
      float qk = threadIdx.x <= i ? (float)qk_buf[threadIdx.x + qk_offset] : 0.0f;
      float left_padding_val = (threadIdx.x < left_padding_len)? -1e20f:0.0f;
      float tmp = (threadIdx.x <= i ) ? (float)(qk * (float)scalar + left_padding_val): -1e20f;

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
        qk_buf[threadIdx.x + qk_offset] = threadIdx.x <= i ?  (T)(qk / s_sum) : (T)0.0f;

      qk_offset += seq_len;
    }
}

template <class T>
void softmax_kernel(void *qk_buf_, const int64_t *__restrict padding_index, const int &batch_size,
                    const int &head_num, const int &seq_len, const float &scalar, const cudaStream_t stream)
{
  dim3 grid, block;

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

  grid.x = batch_size * head_num;
  softmax_kernel_gpt2<T><<<grid, block, 0, stream>>>((T *)qk_buf_, padding_index, head_num, seq_len, scalar);
}

template void softmax_kernel<float>(void *qk_buf_, const int64_t* padding_index,const int& batch_size, 
                                      const int& head_num, const int& seq_len, const float& scalar, const cudaStream_t stream);
template void softmax_kernel<half>(void *qk_buf_, const int64_t* padding_index, const int& batch_size, 
                                      const int& head_num, const int& seq_len, const float& scalar, const cudaStream_t stream);
