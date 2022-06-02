#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "core/common.cuh"

template<typename T>
__global__
void add_bias(T* output, const T* __restrict bias, const int m, const int n)
{
    int id = blockIdx.x * n + blockIdx.y * blockDim.x + threadIdx.x;
    output[id] = output[id] + __ldg(&bias[threadIdx.x + blockDim.x * blockIdx.y]);
}

template <typename T>
__global__ 
void add_bias_input(T* output, const T* input, const T* bias, const int m, const int n)
{
  int id = n * blockIdx.x + blockIdx.y * blockDim.x + threadIdx.x;
  int bias_id = blockIdx.y * blockDim.x + threadIdx.x;
  output[id] = output[id] + input[id] + __ldg(&bias[bias_id]);
}

template <typename T>
__global__ 
void add_input(T* output, const T* input, const int m, const int n)
{
  int id = n * blockIdx.x + blockIdx.y * blockDim.x + threadIdx.x;
  output[id] = output[id] + input[id];
}

template<typename T>
void add_bias_kernel(void* output, const void* bias, const int m, const int n, const cudaStream_t stream)
{
    int fold_coeff = 1;
        if (n <= 1024){
          fold_coeff = 1;
        }else if( n < 2048){
          fold_coeff = 2;
        }else if(n < 4096){
          fold_coeff = 4;
        }else if(n < 8192){
          fold_coeff = 8;
        }else if(n < 16384){
          fold_coeff = 16;
        }

        dim3 grid(m, fold_coeff);
        dim3 block(n / fold_coeff);
        add_bias<<<grid, block, 0, stream>>>((T*)output, (T*)bias,  m, n);
}

template<typename T>
void add_bias_input_kernel(void* output, const void* input, const void* bias,const int m, const int n, const cudaStream_t stream)
{
  int fold_coeff = 1;
  if (n <= 1024){
    fold_coeff = 1;
  }else if( n < 2048){
    fold_coeff = 2;
  }else if(n < 4096){
    fold_coeff = 4;
  }else if(n < 8192){
    fold_coeff = 8;
  }else if(n < 16384){
    fold_coeff = 16;
  }
  dim3 grid(m, fold_coeff);
  dim3 block(n / fold_coeff);
  if (bias != nullptr) {
    add_bias_input<<<grid, block, 0, stream>>>((T*)output, (T*)input, (T*)bias, m, n);
  } else {
    add_input<<<grid, block, 0, stream>>>((T*)output, (T*)input, m, n);
  }
}

template <typename T>
__global__ void add_relative_attn_bias(T *qk_buf, const T *relative_attention_bias, const int seq_len)
{
  int qk_offset = (blockIdx.x + blockIdx.y * gridDim.x) * seq_len;
  int bias_offset = blockIdx.x * seq_len;
  if (threadIdx.x < seq_len) 
    qk_buf[threadIdx.x + qk_offset] += __ldg(&relative_attention_bias[threadIdx.x + bias_offset]);
}

template <class T>
void add_relative_attn_bias_kernel(void *qk_buf, const void *relative_attention_bias, const int &batch_size, const int &head_num, 
                                   const int &seq_len, const cudaStream_t stream)
{
  dim3 block;
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
  
  dim3 grid(head_num * seq_len, batch_size);
  add_relative_attn_bias<T><<<grid, block, 0, stream>>>((T*)qk_buf, (T*)relative_attention_bias, seq_len);
}

template void add_bias_kernel<float>(void* out, const void* bias, const int m, const int n ,const cudaStream_t stream);
template void add_bias_kernel<half>(void* out, const void* bias, const int m, const int n ,const cudaStream_t stream);
template void add_bias_input_kernel<float>(void* output, const void* input, const void* bias,const int m, const int n, const cudaStream_t stream);
template void add_bias_input_kernel<half>(void* output, const void* input, const void* bias,const int m, const int n, const cudaStream_t stream);
template void add_relative_attn_bias_kernel<float>(void *qk_buf, const void *relative_attention_bias, const int &batch_size, const int &head_num, const int &seq_len, const cudaStream_t stream);
template void add_relative_attn_bias_kernel<half>(void *qk_buf, const void *relative_attention_bias, const int &batch_size, const int &head_num, const int &seq_len, const cudaStream_t stream);