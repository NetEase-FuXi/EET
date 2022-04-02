#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
// constants for approximating the normal cdf
// gelu ->gelu_fast
constexpr static float A = 0.5;
constexpr static float B = 0.7978845608028654;   // sqrt(2.0/M_PI)
constexpr static float C = 0.035677408136300125; // 0.044715 * sqrt(2.0/M_PI)
constexpr static float D = 1.702;

template <typename T>
__global__
void add_bias_gelu(T* out, const T* bias, int m, int n)
{
  int idx = n * blockIdx.x + blockIdx.y * blockDim.x + threadIdx.x;
  int bias_idx = blockIdx.y * blockDim.x + threadIdx.x;

  if (idx < m * n){
    T in = out[idx] + bias[bias_idx];
    T cdf = A + A * tanh(in * (C * in * in + B));
    out[idx] = in * cdf;
  }
}

template <>
__global__
void add_bias_gelu<half>(half* out, const half* bias, int m, int n)
{
  const half2 A2 = __floats2half2_rn(A, A);
  const half2 B2 = __floats2half2_rn(B, B);
  const half2 C2 = __floats2half2_rn(C, C);

  half2 * out_ptr = (half2 *)out;
  half2 * bias_ptr = (half2 *)bias;

  int idx = n * blockIdx.x + blockIdx.y * blockDim.x + threadIdx.x;
  int bias_idx = blockIdx.y * blockDim.x + threadIdx.x;

  if (idx < m * n ){
    half2 in = out_ptr[idx] + bias_ptr[bias_idx];
    half2 tmp = in * (C2 * in * in + B2);
    float x = tanh(__half2float(tmp.x));
    float y = tanh(__half2float(tmp.y));
    half2 cdf = A2 + A2 * make_half2(x, y);
    out_ptr[idx] = in * cdf;
  }
}

template <typename T>
__global__
void add_bias_quick_gelu(T* out, const T* bias, int m, int n) 
{
  int idx = n * blockIdx.x + blockIdx.y * blockDim.x + threadIdx.x;
  int bias_idx = blockIdx.y * blockDim.x + threadIdx.x;

  if (idx < m * n) {
    T in = out[idx] + bias[bias_idx];
    T cdf = 1.0f / (1.0f + __expf(-(D * in)));
    out[idx] = in * cdf;
  }
}

template <>
__global__
void add_bias_quick_gelu(half* out, const half* bias, int m, int n)
{
  const half2 D2 = __floats2half2_rn(D, D);
  const half2 half_one2 = __floats2half2_rn(1.0f, 1.0f);

  half2 *out_ptr = (half2 *)out;
  half2 *bias_ptr = (half2 *)bias;

  int idx = n * blockIdx.x + blockIdx.y * blockDim.x + threadIdx.x;
  int bias_idx = blockIdx.y * blockDim.x + threadIdx.x;

  if (idx < m * n) {
    half2 in = out_ptr[idx] + bias_ptr[bias_idx];
    half2 tmp = __hmul2(D2, in);
    float x = __expf(__half2float(-tmp.x));
    float y = __expf(__half2float(-tmp.y));
    half2 cdf = __h2div(half_one2, half_one2 + make_half2(x, y));
    out_ptr[idx] = __hmul2(in, cdf);
  }
}

template <typename T>
__global__
void add_bias_relu(T* out, const T* bias, int m, int n)
{
    int idx = n * blockIdx.x + blockIdx.y * blockDim.x + threadIdx.x;
    int bias_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (idx < m * n){
        T val = out[idx] + bias[bias_idx];
        out[idx] = (T)(val > 0.0f ? val : 0.0f);
    }
}

template <>
  __global__ 
void add_bias_relu(half* out, const half* bias, int m, int n)
{
    int idx =  n * blockIdx.x + blockIdx.y * blockDim.x + threadIdx.x;
    int bias_idx = blockIdx.y * blockDim.x + threadIdx.x;

    half2 *out_ptr = (half2 *)out;
    half2 *bias_ptr = (half2 *)bias;
    if (bias_idx < n && idx < m * n){
        half2 val = __hadd2(out_ptr[idx], bias_ptr[bias_idx]);
        val.x = val.x > (half)0.0f ? val.x : (half)0.0f;
        val.y = val.y > (half)0.0f ? val.y : (half)0.0f;
        out_ptr[idx] = val;
    }
}

template<typename T>
void add_bias_act_kernel(void* ffn_inner, const void* bias, int m, int n ,const int act_type ,const cudaStream_t stream)
{
  if (sizeof(T) == sizeof(half)){

    int fold_coeff = 1;
    if (n <= 2048){
      fold_coeff = 1;
    }else if( n <= 4096){
      fold_coeff = 2;
    }else if(n <= 8192){
      fold_coeff = 4;
    }else if(n <= 16384){
      fold_coeff = 8;
    }else if(n <= 16384 * 2){
      fold_coeff = 16;
    }else if(n <= 16384 * 4){
      fold_coeff = 32;
    }
  
    dim3 grid(m, fold_coeff);
    dim3 block(n / fold_coeff);
  

    block.x /= 2;
    if (act_type == 0) {
      add_bias_relu<T><<<grid, block, 0, stream>>>((T *)ffn_inner, (T *)bias, m, n / 2);
    } else if (act_type == 1) {
      add_bias_gelu<T><<<grid, block, 0, stream>>>((T *)ffn_inner, (T *)bias, m, n / 2);
    } else if (act_type ==2) {
      add_bias_quick_gelu<T><<<grid, block, 0, stream>>>((T *)ffn_inner, (T *)bias, m, n / 2);
    }
    else {
      std::cerr << "unsupported activation " << std::endl;
    }
  } else {

    int fold_coeff = 1;
    if (n <= 1024){
      fold_coeff = 1;
    }else if( n <= 2048){
      fold_coeff = 2;
    }else if(n <= 4096){
      fold_coeff = 4;
    }else if(n <= 8192){
      fold_coeff = 8;
    }else if(n <= 16384){
      fold_coeff = 16;
    }else if(n <= 16384 * 2){
      fold_coeff = 32;
    }else if (n <= 16384 * 4){
      fold_coeff = 64;
    }
  
    dim3 grid(m, fold_coeff);
    dim3 block(n / fold_coeff);
  

    if (act_type == 0) {
      add_bias_relu<T><<<grid, block, 0, stream>>>((T *)ffn_inner, (T *)bias, m, n);
    } else if (act_type == 1) {
      add_bias_gelu<T><<<grid, block, 0, stream>>>((T *)ffn_inner, (T *)bias, m, n);
    } else if (act_type == 2) {
      add_bias_quick_gelu<T><<<grid, block, 0, stream>>>((T *)ffn_inner, (T *)bias, m, n);
    } else {
      std::cerr << "unsupported activation " << std::endl;
    }
  }
}

template void add_bias_act_kernel<float>(void* ffn_inner, const void* bias, const int m, const int n, const int act_type, const cudaStream_t stream);
template void add_bias_act_kernel<half>(void* ffn_inner, const void* bias, const int m, const int n, const int act_type, const cudaStream_t stream);
