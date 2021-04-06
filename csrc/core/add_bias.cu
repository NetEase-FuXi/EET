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
  add_bias_input<<<grid, block, 0, stream>>>((T*)output, (T*)input, (T*)bias, m, n);
}


template void add_bias_kernel<float>(void* out, const void* bias, const int m, const int n ,const cudaStream_t stream);
template void add_bias_kernel<half>(void* out, const void* bias, const int m, const int n ,const cudaStream_t stream);
template void add_bias_input_kernel<float>(void* output, const void* input, const void* bias,const int m, const int n, const cudaStream_t stream);
template void add_bias_input_kernel<half>(void* output, const void* input, const void* bias,const int m, const int n, const cudaStream_t stream);
