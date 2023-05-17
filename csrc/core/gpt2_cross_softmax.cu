#include "core/common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

constexpr int SOFTMAX_BLOCK_SIZE = 128;

template <typename T>
__global__
void cross_softmax_kernel(T *qk_buf_, const int64_t *padding_len, const int head_num, const int seq_len,const int mem_seq_len, const T scalar)
{
  int batch_id = blockIdx.x / (seq_len * head_num);
  int qk_offset = blockIdx.x * mem_seq_len;
  __shared__ float s_sum, s_max;

  int right_padding_len = 0;
  if (padding_len != nullptr)
  {
    right_padding_len = padding_len[batch_id];
  }

  float qk = threadIdx.x < mem_seq_len - right_padding_len ? (float)qk_buf_[threadIdx.x + qk_offset] * (float)scalar : -1e20f;
  float max_val = blockReduceMax<float>(qk);
  if(threadIdx.x == 0)
    s_max = max_val;
  __syncthreads();

  float qk_tmp = threadIdx.x < mem_seq_len - right_padding_len ? __expf((float)(qk - s_max)) : 0.0f;
  float sum_val = blockReduceSum<float>(qk_tmp);

  if(threadIdx.x == 0)
  {
    s_sum = sum_val + 1e-6f;
  }
  __syncthreads();

  if(threadIdx.x < mem_seq_len)
    qk_buf_[threadIdx.x + qk_offset] = (T)(qk_tmp / s_sum);
}

template <typename T, int block_size>
__global__ void cross_softmax_kernel_v2(T *qk_buf_, const int64_t *padding_len, const int head_num, const int seq_len, const int mem_seq_len, const T scalar)
{
  const int tid = threadIdx.x;
  int batch_id = blockIdx.x / (seq_len * head_num);
  int qk_offset = blockIdx.x * mem_seq_len;

  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<float*>(shared_buf);
  __shared__ float s_sum, s_max;

  int right_padding_len = 0;
  if (padding_len != nullptr)
  {
    right_padding_len = padding_len[batch_id];
  }

  // load & max
  float thread_max = -1e20f;
  for (int col_id = tid; col_id < mem_seq_len - right_padding_len; col_id += block_size) {
    buf[col_id] = static_cast<float>(qk_buf_[qk_offset + col_id]) * (float)scalar;
    thread_max = max(thread_max, buf[col_id]);
  }
  float max_val = blockReduceMax<float>(thread_max);
  if (threadIdx.x == 0)
    s_max = max_val;
  __syncthreads();

  // sum
  float thread_sum = 0;
  for (int col_id = tid; col_id < mem_seq_len - right_padding_len; col_id += block_size) {
    float exp_x = __expf(buf[col_id] - s_max);
    buf[col_id] = exp_x;
    thread_sum += exp_x;
  }
  float sum_val = blockReduceSum<float>(thread_sum);

  if (threadIdx.x == 0)
  {
    s_sum = sum_val + 1e-6f;
  }
  __syncthreads();

  // store
  for (int col_id = tid; col_id < mem_seq_len; col_id += block_size) {
    qk_buf_[qk_offset + col_id] = col_id < mem_seq_len - right_padding_len ? (T)(buf[col_id] / s_sum) : (T)0.0f;
  }
}

template <class T>
void launch_cross_softmax_kernel(void *qk_buf_, const int64_t *padding_len, const int &batch_size,
                                 const int &head_num, const int &seq_len, const int &mem_seq_len, const float &scalar, const cudaStream_t stream)
{
  const size_t smem_size = mem_seq_len * sizeof(float);
  const int grid_dim_x = batch_size * head_num * seq_len;
  int block_dim_x;

  assert(mem_seq_len <= 2048);
  if (seq_len <= 128) {
    block_dim_x = min(((mem_seq_len + 31) / 32) * 32, 1024);
    cross_softmax_kernel<T><<<grid_dim_x, block_dim_x, 0, stream>>>((T *)qk_buf_, padding_len, head_num, seq_len, mem_seq_len, scalar);
  } else {
    block_dim_x = SOFTMAX_BLOCK_SIZE;
    cross_softmax_kernel_v2<T, SOFTMAX_BLOCK_SIZE><<<grid_dim_x, block_dim_x, smem_size, stream>>>((T *)qk_buf_, padding_len, head_num, seq_len, mem_seq_len, scalar);
  }
}

template void launch_cross_softmax_kernel<float>(void *qk_buf_, const int64_t *padding_len, const int& batch_size,
                                                 const int& head_num, const int& seq_len,const int& mem_seq_len, const float& scalar, const cudaStream_t stream);

template void launch_cross_softmax_kernel<half>(void *qk_buf_, const int64_t *padding_len, const int& batch_size,
                                                const int& head_num, const int& seq_len,const int& mem_seq_len, const float& scalar, const cudaStream_t stream);

