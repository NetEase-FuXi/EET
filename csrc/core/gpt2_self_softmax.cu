#include "core/common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void softmax_kernel_gpt2(T *qk_buf_, const int64_t *__restrict padding_index, const int head_num, const int seq_len, const T scalar)
{
  int batch_id = blockIdx.x / head_num;
  int qk_offset = blockIdx.x * seq_len * seq_len;

  __shared__ float s_sum, s_max;

  int left_padding_len = 0;
  if (padding_index != nullptr)
  {
    left_padding_len = padding_index[batch_id];
  }
  qk_offset += left_padding_len * seq_len;
  for (int i = left_padding_len; i < seq_len; ++i)
  {
    if (threadIdx.x + blockDim.x * blockIdx.y < seq_len)
    {
      float qk = (float)qk_buf_[threadIdx.x + blockDim.x * blockIdx.y+ qk_offset];

      float tmp = (threadIdx.x+ blockDim.x * blockIdx.y <= i && threadIdx.x+ blockDim.x * blockIdx.y >= left_padding_len) ? (float)(qk * (float)scalar) : -1e20f;

      float max_val = blockReduceMax<float>(tmp);

      if (threadIdx.x == 0)
        s_max = max_val;
      __syncthreads();

      qk = __expf(tmp - s_max);

      float sum_val = blockReduceSum<float>(qk);

      if (threadIdx.x == 0)
      {
        s_sum = sum_val + 1e-6f;
      }
      __syncthreads();

      qk_buf_[threadIdx.x+ blockDim.x * blockIdx.y + qk_offset] = (T)(qk / s_sum);

      qk_offset += seq_len;
    }
  }
}

template <class T>
void softmax_kernel(void *qk_buf_, const int64_t* __restrict padding_index, const int& batch_size, 
                    const int& head_num, const int& seq_len, const float& scalar, const cudaStream_t stream){
  dim3 grid, block;

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

    int fold_coeff = 1;
    if (seq_len <= 1024){
        fold_coeff = 1;
    }else if(seq_len <= 2048){
        fold_coeff = 2;
    }else if(seq_len <= 4096){
        fold_coeff = 4;
    }else if(seq_len <= 8192){
        fold_coeff = 8;
    }else if(seq_len <= 16384){
        fold_coeff = 16;
    }
    // support lager seq_len
    grid.x = batch_size * head_num;
    grid.y = fold_coeff;
    softmax_kernel_gpt2 <T><<<grid, block, 0, stream>>>((T*)qk_buf_, padding_index, head_num, seq_len, scalar);

  }

template void softmax_kernel<float>(void *qk_buf_, const int64_t* padding_index,const int& batch_size, 
                                      const int& head_num, const int& seq_len, const float& scalar, const cudaStream_t stream);
template void softmax_kernel<half>(void *qk_buf_, const int64_t* padding_index, const int& batch_size, 
                                      const int& head_num, const int& seq_len, const float& scalar, const cudaStream_t stream);