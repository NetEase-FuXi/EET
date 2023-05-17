#include "core/common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr int SOFTMAX_BLOCK_SIZE = 128;
constexpr int WARP_SIZE = 32;

template <typename T>
__global__ void bert_softmax_kernel(T *qk_buf, const int64_t *padding_len, const int head_num, const int seq_len)
{
  int batch_id = blockIdx.x / (seq_len * head_num);
  int qk_offset = blockIdx.x * seq_len;
  __shared__ float s_sum, s_max;
  
  int right_padding_len = 0;
  if (padding_len != nullptr)
  {
    right_padding_len = padding_len[batch_id];
  }

  float qk = threadIdx.x < seq_len - right_padding_len ? static_cast<float>(qk_buf[threadIdx.x + qk_offset]) : -1e20f;
  float max_val = blockReduceMax<float>(qk);
  if (threadIdx.x == 0)
    s_max = max_val;
  __syncthreads();

  float qk_tmp = threadIdx.x < seq_len - right_padding_len ? __expf((float)(qk - s_max)) : 0.0f;
  float sum_val = blockReduceSum<float>(qk_tmp);

  if (threadIdx.x == 0)
  {
    s_sum = sum_val + 1e-6f;
  }
  __syncthreads();

  if (threadIdx.x < seq_len)
    qk_buf[threadIdx.x + qk_offset] = (T)(qk_tmp / s_sum);
}

template <typename T, int block_size>
__global__ void bert_softmax_kernel_v2(T *qk_buf, const int64_t *padding_len, const int head_num, const int seq_len)
{
  const int tid = threadIdx.x;
  int batch_id = blockIdx.x / (seq_len * head_num);
  int qk_offset = blockIdx.x * seq_len;

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
  for (int col_id = tid; col_id < seq_len - right_padding_len; col_id += block_size) {
    buf[col_id] = static_cast<float>(qk_buf[qk_offset + col_id]);
    thread_max = max(thread_max, buf[col_id]);
  }
  float max_val = blockReduceMax<float>(thread_max);
  if (threadIdx.x == 0)
    s_max = max_val;
  __syncthreads();

  // sum
  float thread_sum = 0;
  for (int col_id = tid; col_id < seq_len - right_padding_len; col_id += block_size) {
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
  for (int col_id = tid; col_id < seq_len; col_id += block_size) {
    qk_buf[qk_offset + col_id] = col_id < seq_len - right_padding_len ? (T)(buf[col_id] / s_sum) : (T)0.0f;
  }
}

template <typename T>
__global__ void masked_softmax_kernel(T *qk_buf, const int64_t *padding_len, const int head_num, const int seq_len)
{
  int batch_id = blockIdx.x / head_num;
  int qk_offset = blockIdx.x * seq_len * seq_len;
  int mask_offset = batch_id * seq_len * seq_len;
  __shared__ float s_sum, s_max;

  int right_padding_len = 0;
  if (padding_len != nullptr)
  {
    right_padding_len = padding_len[batch_id];
  }

  for (int i = 0; i < seq_len - right_padding_len; ++i)
  {
    float qk = threadIdx.x < seq_len ? static_cast<float>(qk_buf[threadIdx.x + qk_offset]) : 0.0f;
    float padding_val = (threadIdx.x > i || threadIdx.x >= seq_len - right_padding_len) ? -1e20f : 0.0f;

    float tmp = threadIdx.x < seq_len ? (float)(qk + padding_val) : -1e20f;
    float max_val = blockReduceMax<float>(tmp);

    if (threadIdx.x == 0)
      s_max = max_val;
    __syncthreads();

    qk = threadIdx.x < seq_len ? __expf(tmp - s_max) : 0.0f;

    float sum_val = blockReduceSum<float>(qk);

    if (threadIdx.x == 0)
    {
      s_sum = sum_val + 1e-6f;
    }
    __syncthreads();

    if (threadIdx.x < seq_len)
      qk_buf[threadIdx.x + qk_offset] = (T)(qk / s_sum);

    qk_offset += seq_len;
  }

  for (int i = seq_len - right_padding_len; i < seq_len; ++i)
  {
    if (threadIdx.x < seq_len)
      qk_buf[threadIdx.x + qk_offset] = (T)0.0f;
    qk_offset += seq_len;
  }
}

template <typename T>
__global__ void t5_softmax_kernel(T *qk_buf, T *position_bias, const int64_t *padding_len, const int head_num, const int seq_len)
{
  int batch_id = blockIdx.x / (seq_len * head_num);
  int qk_offset = blockIdx.x * seq_len;
  int bias_offset = blockIdx.x % (seq_len * head_num) * seq_len;
  __shared__ float s_sum, s_max;

  int right_padding_len = 0;
  if (padding_len != nullptr)
  {
    right_padding_len = padding_len[batch_id];
  }

  float qk = threadIdx.x < seq_len - right_padding_len ? static_cast<float>(qk_buf[threadIdx.x + qk_offset]) + static_cast<float>(position_bias[threadIdx.x + bias_offset]) : -1e20f;
  float max_val = blockReduceMax<float>(qk);
  if (threadIdx.x == 0)
    s_max = max_val;
  __syncthreads();

  float qk_tmp = threadIdx.x < seq_len - right_padding_len ? __expf((float)(qk - s_max)) : 0.0f;
  float sum_val = blockReduceSum<float>(qk_tmp);

  if (threadIdx.x == 0)
  {
    s_sum = sum_val + 1e-6f;
  }
  __syncthreads();

  if (threadIdx.x < seq_len)
    qk_buf[threadIdx.x + qk_offset] = (T)(qk_tmp / s_sum);
}

template <typename T, int block_size>
__global__ void t5_softmax_kernel_v2(T *qk_buf, T *position_bias, const int64_t *padding_len, const int head_num, const int seq_len)
{
  const int tid = threadIdx.x;
  int batch_id = blockIdx.x / (seq_len * head_num);
  int qk_offset = blockIdx.x * seq_len;
  int bias_offset = blockIdx.x % (seq_len * head_num) * seq_len;

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
  for (int col_id = tid; col_id < seq_len - right_padding_len; col_id += block_size) {
    buf[col_id] = static_cast<float>(qk_buf[qk_offset + col_id]) + static_cast<float>(position_bias[bias_offset + col_id]);
    thread_max = max(thread_max, buf[col_id]);
  }
  float max_val = blockReduceMax<float>(thread_max);
  if (threadIdx.x == 0)
    s_max = max_val;
  __syncthreads();

  // sum
  float thread_sum = 0;
  for (int col_id = tid; col_id < seq_len - right_padding_len; col_id += block_size) {
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
  for (int col_id = tid; col_id < seq_len; col_id += block_size) {
    qk_buf[qk_offset + col_id] = col_id < seq_len - right_padding_len ? (T)(buf[col_id] / s_sum) : (T)0.0f;
  }
}

template <class T>
void launch_softmax_kernel(void *qk_buf, void* position_bias, const int64_t *padding_len, const int batch_size, const int head_num,
                              const int seq_len, bool need_sequence_mask, const cudaStream_t stream)
{
  const size_t smem_size = seq_len * sizeof(float);
  const int grid_dim_x = batch_size * head_num * seq_len;
  int block_dim_x;

  assert(seq_len <= 2048);
  if (need_sequence_mask) {
    block_dim_x = min(((seq_len + 31) / 32) * 32, 1024);
    masked_softmax_kernel<T><<<grid_dim_x, block_dim_x, 0, stream>>>((T*)qk_buf, padding_len, head_num, seq_len);
  } else {
    if (position_bias == nullptr) {
      if (seq_len <= 128) {
        block_dim_x = min(((seq_len + 31) / 32) * 32, 1024);
        bert_softmax_kernel<T><<<grid_dim_x, block_dim_x, 0, stream>>>((T*)qk_buf, padding_len, head_num, seq_len);
      } else {
        block_dim_x = SOFTMAX_BLOCK_SIZE;
        bert_softmax_kernel_v2<T, SOFTMAX_BLOCK_SIZE><<<grid_dim_x, block_dim_x, smem_size, stream>>>((T *)qk_buf, padding_len, head_num, seq_len);
      }
    } else {
      if (seq_len <= 128) {
        block_dim_x = min(((seq_len + 31) / 32) * 32, 1024);
        t5_softmax_kernel<T><<<grid_dim_x, block_dim_x, 0, stream>>>((T *)qk_buf, (T*)position_bias, padding_len, head_num, seq_len);
      } else {
        block_dim_x = SOFTMAX_BLOCK_SIZE;
        t5_softmax_kernel_v2<T, SOFTMAX_BLOCK_SIZE><<<grid_dim_x, block_dim_x, smem_size, stream>>>((T *)qk_buf, (T*)position_bias, padding_len, head_num, seq_len);
      }
    }
  }
}

template void launch_softmax_kernel<float>(void *qk_buf, void *position_bias, const int64_t *padding_len, const int batch_size,
                                           const int head_num, const int seq_len, bool need_sequence_mask, const cudaStream_t stream);
template void launch_softmax_kernel<half>(void *qk_buf, void *position_bias, const int64_t *padding_len, const int batch_size,
                                          const int head_num, const int seq_len, bool need_sequence_mask, const cudaStream_t stream);
