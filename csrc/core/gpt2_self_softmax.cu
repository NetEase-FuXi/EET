#include <cuda_fp16.h>
#include <iostream>
#include <cuda_runtime.h>
#include "core/common.cuh"

template <typename T>
__global__
void softmax_kernel_gpt2(T *qk_buf,const int64_t* __restrict padding_index, const int head_num, const int seq_len)
{
    int batch_id = blockIdx.x / head_num;
    int qk_offset = blockIdx.x * seq_len * seq_len;

    __shared__ float s_sum, s_max;
    
    int left_padding_len = 0;
    if (padding_index != nullptr){
        left_padding_len = padding_index[batch_id];
    }

    // To avoid overflow
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
      float tmp = (threadIdx.x <= i ) ? (float)(qk + left_padding_val): -1e20f;

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

template <typename T,int item_per_thread>
__global__
void softmax_kernel_gpt2_opt(T *qk_buf,const int64_t* __restrict padding_index, const int head_num, const int seq_len)
{
    int batch_id = blockIdx.x / head_num;
    int qk_offset = blockIdx.x * seq_len * seq_len;
    int thread_id = threadIdx.x;
    __shared__ float s_sum, s_max;
    
    int left_padding_len = 0;
    if (padding_index != nullptr){
        left_padding_len = padding_index[batch_id];
    }

    // To avoid overflow
    for (int i = 0; i < left_padding_len; i++)
    {
        for (int j = 0; j < item_per_thread; j++){
          if(thread_id * item_per_thread + j < seq_len)
            qk_buf[thread_id * item_per_thread + j + qk_offset] = (T)0.0f;
            }
      qk_offset += seq_len;
    }
    

    float qk[item_per_thread];
    float left_padding_val[item_per_thread];
    float tmp[item_per_thread];

    for(int i = left_padding_len; i < seq_len; ++i)
    {
        for (int j = 0; j < item_per_thread; j++){
            qk[j] =(thread_id * item_per_thread + j) <= i ? (float)qk_buf[thread_id * item_per_thread + j + qk_offset] : 0.0f;
            left_padding_val[j] = ((thread_id * item_per_thread + j) < left_padding_len)? -1e20f:0.0f;
            tmp[j] = ((thread_id * item_per_thread + j) <= i ) ? (float)(qk[j] + left_padding_val[j]): -1e20f;
        }

        float max_val = blockReduceMax_opt<float,item_per_thread>(tmp);

        if(thread_id == 0)
            s_max = max_val;
        __syncthreads();

        for (int j = 0; j < item_per_thread; j++){
            qk[j] = (thread_id * item_per_thread + j) <= i ? __expf(tmp[j] - s_max) : 0.0f;
        }
        float sum_val = blockReduceSum_opt<float,item_per_thread>(qk);

        if(thread_id == 0)
        {
            s_sum = sum_val + 1e-6f;
        }
        __syncthreads();


        for (int j = 0; j < item_per_thread; j++){
            if((thread_id * item_per_thread + j) < seq_len)
                qk_buf[thread_id * item_per_thread + j + qk_offset] = thread_id * item_per_thread + j <= i ?  (T)(qk[j] / s_sum) : (T)0.0f;
        }
        qk_offset += seq_len;
    }
}

template <class T>
void softmax_kernel(void *qk_buf_, const int64_t *__restrict padding_index, const int &batch_size,
                    const int &head_num, const int &seq_len, const cudaStream_t stream)
{
  dim3 grid, block;

  if(seq_len <= 1024)
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
    grid.x = batch_size * head_num;

    softmax_kernel_gpt2<T><<<grid, block, 0, stream>>>((T *)qk_buf_, padding_index, head_num, seq_len);
  }
  else
  {
    grid.x = batch_size * head_num;
    if (seq_len <= 2048)
    {
      // block.x = seq_len/2;
      block.x = ceil(seq_len / (32.0 * 2)) * 32;         // item_per_thread = 1
      softmax_kernel_gpt2_opt<T,2><<<grid, block, 0, stream>>>((T *)qk_buf_, padding_index, head_num, seq_len);
    }
    else if (seq_len <= 4096)
    {
      block.x = ceil(seq_len / (32.0 * 4)) * 32;         // item_per_thread = 1
      softmax_kernel_gpt2_opt<T,4><<<grid, block, 0, stream>>>((T *)qk_buf_, padding_index, head_num, seq_len);
    }
    else
    {
      std::cerr << "not support seq_len for softmax" << std::endl;
    }
    // else if (seq_len <= 8192)
    // {
    //   block.x = ceil(seq_len / (32.0 * 8)) * 32;         // item_per_thread = 1
    //   softmax_kernel_gpt2_opt<T,8><<<grid, block, 0, stream>>>((T *)qk_buf_, padding_index, head_num, seq_len);
    // }
    // else if (seq_len <= 16384)
    // {
    //   block.x = ceil(seq_len / (32.0 * 16)) * 32;         // item_per_thread = 1
    //   softmax_kernel_gpt2_opt<T,16><<<grid, block, 0, stream>>>((T *)qk_buf_, padding_index, head_num, seq_len);
    // }
  }
}

template void softmax_kernel<float>(void *qk_buf_, const int64_t* padding_index,const int& batch_size, 
                                      const int& head_num, const int& seq_len, const cudaStream_t stream);
template void softmax_kernel<half>(void *qk_buf_, const int64_t* padding_index, const int& batch_size, 
                                      const int& head_num, const int& seq_len, const cudaStream_t stream);
