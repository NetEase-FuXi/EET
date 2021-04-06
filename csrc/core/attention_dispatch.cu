#include "core/common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <assert.h>
#include "cub/cub.cuh"

const int WARP_SIZE = 32;
const bool ATTENION_OPT = true;
const int ATTENTION_BLOCK_SIZE = 256;

// attention_dispatch code modified from Nvidia's DeepLearningExamples
// https://github.com/NVIDIA/DeepLearningExamples/blob/master/FasterTransformer/v3.1/fastertransformer/cuda/open_decoder.cu#L194-L824

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int HALF_ELEMENTS_PER_WARP_LOAD>
using Copy_half_t =
    typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 32, half,
        typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 64, int,
            typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 128, int2, int4
            >::type
        >::type
    >::type;

template <typename T, int ELEMENTS_PER_WARP_LOAD>
using Copy_t = Copy_half_t<sizeof(T) / sizeof(half) * ELEMENTS_PER_WARP_LOAD>;

///////////////////////////////////////////////////////////////////////////////////////////////////

#if 1
template <typename T>
__global__ 
void masked_attention_kernel(
  T* key_buf, T* value_buf,
  T* query_buf, const T* self_Q_bias, 
  T* key_cache, const T* self_K_bias, T* value_cache, const T* self_V_bias,
  T* context_buf, int batch_size, int head_num, int size_per_head, const int step, const T scalar, int* padding_index)
{
  extern __shared__ __align__(sizeof(T)) unsigned s_buf[];
  T* sq = reinterpret_cast<T *>(s_buf);
  T* logits = reinterpret_cast<T *>(&sq[size_per_head]);

  int tid = threadIdx.x;
  int bid = blockIdx.x / head_num;
  int batch_id = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head + tid;
  int qkv_bias_id = head_id * size_per_head + tid;


  if(tid < size_per_head)
    sq[tid] = query_buf[qkv_id] + self_Q_bias[qkv_bias_id];
  __syncthreads();

  //offset for each step
  int offset = batch_size * head_num * size_per_head;
  for(int ite = 0; ite < step; ++ite)
  {
    T key = tid < size_per_head ? key_cache[ite * offset + qkv_id] : (T)0.0f;
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1 && tid < size_per_head)
    {
      key = key_buf[qkv_id] + self_K_bias[qkv_bias_id];
      key_cache[ite * offset + qkv_id] = key;
    }


    T val = (tid < size_per_head) ? key * sq[tid] * scalar : (T)(0.0f);
    T qk = blockReduceSum(val);
    if(threadIdx.x == 0){
      int padding_len = 0;
      if (padding_index != nullptr){
          padding_len = padding_index[batch_id];
      }
      logits[ite] = ite >= padding_len ? qk : (T)-1e20f;
    }
    __syncthreads(); //try to remove
  }
  __syncthreads(); //try to remove

  __shared__ float s_max_val, s_sum;
  float local_i = tid < step ? (float)logits[tid] : -1e20f; 
  float max_val = blockReduceMax<float>(local_i);
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  local_i -= s_max_val;
  float local_o = tid < step ? __expf(local_i) : 0.0f;
  float val = blockReduceSum<float>(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();

  if(tid < step)
    logits[tid] = local_o / s_sum;
  __syncthreads();

    if(tid < size_per_head)
  {
    T sum = (T)0.0f;
    for(int ite = 0; ite < step; ++ite)
    {
      T value = value_cache[ite * offset + qkv_id];
      //for the last step, we should update K + bias_K to the cache
      if(ite == step - 1)
      {
        value = value_buf[qkv_id] + self_V_bias[qkv_bias_id];
        value_cache[ite * offset + qkv_id] = value;
      }
      sum += value * logits[ite];
    }
    context_buf[qkv_id] = sum;
  }
}

// only use for compile 
template <int size_per_head, int block_sz>
__global__ 
void masked_attention_kernel_opt_half2(
  float* __restrict key_buf, float* __restrict value_buf,
  float* __restrict query_buf, const float* __restrict self_Q_bias, 
  float* __restrict key_cache, const float* __restrict self_K_bias, 
  float* __restrict value_cache, const float* __restrict self_V_bias,
  float* __restrict context_buf, int batch_size, int head_num, const int step, const float scalar, int* padding_index) {}

template <int size_per_head, int block_sz>
__global__ 
void masked_attention_kernel_opt_half2(
  half* __restrict key_buf, half* __restrict value_buf,
  half* __restrict query_buf, const half* __restrict self_Q_bias, 
  half* __restrict key_cache, const half* __restrict self_K_bias, 
  half* __restrict value_cache, const half* __restrict self_V_bias,
  half* __restrict context_buf, int batch_size, int head_num, const int step, const half scalar, int* padding_index)
{
  half2* key_buf_ptr = (half2*)key_buf;
  half2* value_buf_ptr = (half2*)value_buf;
  half2* query_buf_ptr = (half2*)query_buf;
  half2* key_cache_ptr = (half2*)key_cache;
  half2* value_cache_ptr = (half2*)value_cache;
  const half2* self_Q_bias_ptr = (const half2*)self_Q_bias;
  const half2* self_K_bias_ptr = (const half2*)self_K_bias;
  const half2* self_V_bias_ptr = (const half2*)self_V_bias;
  half2* context_buf_ptr = (half2*)context_buf;

  typedef Copy_t<half2, size_per_head/2> copy_t;
  const int elems_per_thread = size_per_head / 2 / WARP_SIZE;

  union Access_t
  {
    copy_t v;
    half2 x[elems_per_thread]; // supported size 1,2,4
  };
  typedef struct Half_n_t
  {
    half2 x[elems_per_thread]; // supported size 1,2,4
  } half_n_t;

  __shared__ half_n_t sq[block_sz];

  __shared__ float logits[1024]; // only use [0 ~ step-1]

  const int tid = threadIdx.x;
  const int warp_num = block_sz / WARP_SIZE;
  const int bid = blockIdx.x;
  const int batch_id = bid / head_num;
  const int head_id = blockIdx.x % head_num;
  const int warp_id = tid / WARP_SIZE; // warp_id in block
  const int lane_id = tid % WARP_SIZE; // lane_id in warp

  typedef cub::BlockReduce<float, block_sz> MaxValBlockReduce;
  typedef cub::BlockReduce<float, block_sz> BlockReduce;
  __shared__ typename MaxValBlockReduce::TempStorage max_val_block_temp_storage;
  __shared__ typename BlockReduce::TempStorage block_temp_storage;
  __shared__ typename cub::WarpReduce<float>::TempStorage temp_storage[warp_num];

  int qkv_id = bid * size_per_head / 2;
  int qkv_bias_id = head_id * size_per_head / 2;

  query_buf_ptr = &query_buf_ptr[qkv_id];
  key_buf_ptr = &key_buf_ptr[qkv_id];
  value_buf_ptr = &value_buf_ptr[qkv_id];
  self_K_bias_ptr = &self_K_bias_ptr[qkv_bias_id];
  key_cache_ptr = &key_cache_ptr[qkv_id];
  self_Q_bias_ptr = &self_Q_bias_ptr[qkv_bias_id];
  self_V_bias_ptr = &self_V_bias_ptr[qkv_bias_id];
  value_cache_ptr = &value_cache_ptr[qkv_id];
  context_buf_ptr = &context_buf_ptr[qkv_id];

  Access_t bias_r, query_buf_r;
  Access_t key_val_r, key_buf_r;
  Access_t value_val_r, value_buf_r;

  // each warp will have its own copy of sq
  query_buf_r.v = *((copy_t *)query_buf_ptr + lane_id);
  key_buf_r.v = *((copy_t *)key_buf_ptr + lane_id);
  bias_r.v = *((copy_t *)self_Q_bias_ptr + lane_id);
  half2 qb_r[elems_per_thread];
  for (int i = 0; i < elems_per_thread; ++i)
  {
    qb_r[i] = __hadd2(query_buf_r.x[i], bias_r.x[i]);
  }

  //offset for each step
  int offset = batch_size * head_num * size_per_head / 2;
  bias_r.v = *((copy_t *) self_K_bias + lane_id);
  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    key_val_r.v = *((copy_t *)&key_cache_ptr[ite * offset] + lane_id);
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        key_val_r.x[i] = __hadd2(key_buf_r.x[i], bias_r.x[i]);
      }
      *((copy_t *)&key_cache_ptr[ite * offset] + lane_id) = key_val_r.v;
    }
    float val = 0.f;
    for (int i = 0; i < elems_per_thread; i++)
    {
      half2 val2 = __hmul2(key_val_r.x[i], qb_r[i]);
      val = val + (float)((val2.x + val2.y) * scalar);
    }
    float qk = cub::WarpReduce<float>(temp_storage[warp_id]).Sum(val);
    if (lane_id == 0)
    {
      int padding_len = 0;
      if (padding_index != nullptr){
          padding_len = padding_index[batch_id];
          // if (padding_len>0)
          // {
          //   printf("batch_id:%d  padding_len:%d\n",batch_id,padding_len);
          // }
          
      }
      logits[ite] = ite >= padding_len ? qk : -1e20f;
    }
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;
  float local_i = -1e20f;
  for(int i = tid; i < step; i += blockDim.x)
    local_i = max(local_i, logits[i]);

  float max_val = MaxValBlockReduce(max_val_block_temp_storage).Reduce(local_i, cub::Max());
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  float local_o = 0.0f;
  for(int i = tid; i < step; i += blockDim.x)
  {
    logits[i] = __expf(logits[i] - s_max_val);
    local_o += logits[i];
  }
  float val = BlockReduce(block_temp_storage).Sum(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();

  float s_sum_inverse = __fdividef(1.0f, s_sum);
  for(int i = tid; i < step; i += blockDim.x)
  {
    logits[i] = logits[i] * s_sum_inverse;
  }
  __syncthreads(); 

  // This optimization introduces discrepancy because of different order in FP32 summation
  half2 sum_r[elems_per_thread];
  for(int i = 0; i < elems_per_thread; i++)
  {
    sum_r[i].x = (half)0.f;
    sum_r[i].y = (half)0.f;
  }
  bias_r.v = *((copy_t *) self_V_bias_ptr + lane_id);
  value_buf_r.v = *((copy_t *)value_buf_ptr + lane_id);

  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    value_val_r.v = *((copy_t *)&value_cache_ptr[ite * offset] + lane_id);
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        value_val_r.x[i] = __hadd2(value_buf_r.x[i], bias_r.x[i]);
      }
      *((copy_t *)&value_cache_ptr[ite * offset] + lane_id) = value_val_r.v;
    }
    for (int i = 0; i < elems_per_thread; ++i)
    {
      half2 logit2_val;
      logit2_val.x = (half)logits[ite];
      logit2_val.y = (half)logits[ite];
      sum_r[i] = __hadd2(sum_r[i], __hmul2(value_val_r.x[i], logit2_val));
    }
  }
  for (int i = 0; i < elems_per_thread; i++)
  {
    sq[warp_id * WARP_SIZE + lane_id].x[i] = sum_r[i];
  }
  __syncthreads();
  if (warp_id == 0)
  {
    #pragma unroll
    for (int j = 1; j < warp_num; j++)
    {
      for (int i = 0; i < elems_per_thread; ++i)
      {
        sum_r[i] = __hadd2(sum_r[i], sq[j * WARP_SIZE + tid].x[i]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < elems_per_thread; i++)
  {
    value_val_r.x[i] = sum_r[i];
  }
  if (warp_id == 0)
  {
    *((copy_t *)context_buf_ptr + lane_id) = value_val_r.v;
  }
}

template <int size_per_head, int block_sz, typename T>
__global__ 
void masked_attention_kernel_opt(
  T* __restrict key_buf, T* __restrict value_buf,
  T* __restrict query_buf, const T* __restrict self_Q_bias, 
  T* __restrict key_cache, const T* __restrict self_K_bias, 
  T* __restrict value_cache, const T* __restrict self_V_bias,
  T* __restrict context_buf, int batch_size, int head_num, const int step, const T scalar, int* padding_index)
{
  typedef Copy_t<T, size_per_head> copy_t;
  const int elems_per_thread = size_per_head / WARP_SIZE;

  union Access_t
  {
    copy_t v;
    T x[elems_per_thread]; // supported size 1,2,4
  };
  typedef struct Float_n_t
  {
    T x[elems_per_thread]; // supported size 1,2,4
  } float_n_t;

  __shared__ float_n_t sq[block_sz];

  __shared__ float logits[1024]; // only use [0 ~ step-1], the step should be smaller than 1024

  const int tid = threadIdx.x;
  const int warp_num = block_sz / WARP_SIZE;
  const int bid = blockIdx.x;
  const int batch_id = blockIdx.x / head_num;
  const int head_id = blockIdx.x % head_num;
  const int warp_id = tid / WARP_SIZE; // warp_id in block
  const int lane_id = tid % WARP_SIZE; // lane_id in warp

  typedef cub::BlockReduce<float, block_sz> MaxValBlockReduce;
  typedef cub::BlockReduce<float, block_sz> BlockReduce;
  __shared__ typename MaxValBlockReduce::TempStorage max_val_block_temp_storage;
  __shared__ typename BlockReduce::TempStorage block_temp_storage;
  __shared__ typename cub::WarpReduce<float>::TempStorage temp_storage[warp_num];

  int qkv_id = bid * size_per_head;
  int qkv_bias_id = head_id * size_per_head;

  query_buf = &query_buf[qkv_id];
  key_buf = &key_buf[qkv_id];
  value_buf = &value_buf[qkv_id];
  self_K_bias = &self_K_bias[qkv_bias_id];
  key_cache = &key_cache[qkv_id];
  self_Q_bias = &self_Q_bias[qkv_bias_id];
  self_V_bias = &self_V_bias[qkv_bias_id];
  value_cache = &value_cache[qkv_id];
  context_buf = &context_buf[qkv_id];

  Access_t bias_r, query_buf_r;
  Access_t key_val_r, key_buf_r;
  Access_t value_val_r, value_buf_r;

  // each warp will have its own copy of sq
  query_buf_r.v = *((copy_t *)query_buf + lane_id);
  key_buf_r.v = *((copy_t *)key_buf + lane_id);
  bias_r.v = *((copy_t *)self_Q_bias + lane_id);
  float qb_r[elems_per_thread];
  for (int i = 0; i < elems_per_thread; ++i)
  {
    qb_r[i] =  (float)query_buf_r.x[i] + (float)bias_r.x[i];
  }

  //offset for each step
  int offset = batch_size * head_num * size_per_head;
  bias_r.v = *((copy_t *) self_K_bias + lane_id);
  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    key_val_r.v = *((copy_t *)&key_cache[ite * offset] + lane_id);
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        key_val_r.x[i] = (float)key_buf_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&key_cache[ite * offset] + lane_id) = key_val_r.v;
      //*((copy_t *)&key_cache_t[ite * size_per_head] + lane_id) = key_val_r.v;
    }
    float val = 0.f;
    for (int i = 0; i < elems_per_thread; i++)
    {
      val = val +  (float)key_val_r.x[i] * qb_r[i] * (float)scalar;
    }
    float qk = cub::WarpReduce<float>(temp_storage[warp_id]).Sum(val);
    if (lane_id == 0)
    {
      int padding_len = 0;
      if (padding_index != nullptr){
          padding_len = padding_index[batch_id];
          // printf("!!!!!!!!!!!!!!!!!!!!!!!!!  %d\n",padding_len);
      }
      logits[ite] = ite >= padding_len ? (T)qk : (T)-1e20f;
      //logits[ite] = qk;
    }
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;

  float local_i = -1e20f;
  for(int i = tid; i < step; i += blockDim.x)
    local_i = max(local_i, logits[i]);

  float max_val = MaxValBlockReduce(max_val_block_temp_storage).Reduce(local_i, cub::Max());
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();


  float local_o = 0.0f;
  for(int i = tid; i < step; i += blockDim.x)
  {
    logits[i] = __expf(logits[i] - s_max_val);
    local_o += logits[i];
  }
  float val = BlockReduce(block_temp_storage).Sum(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();

  float s_sum_inverse = __fdividef(1.0f, s_sum);
  for(int i = tid; i < step; i += blockDim.x)
  {
    logits[i] = logits[i] * s_sum_inverse;
  }
  __syncthreads(); 

  // This optimization introduces discrepancy because of different order in FP32 summation
  float sum_r[elems_per_thread] = {0.f};
  bias_r.v = *((copy_t *) self_V_bias + lane_id);
  value_buf_r.v = *((copy_t *)value_buf + lane_id);

  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    value_val_r.v = *((copy_t *)&value_cache[ite * offset] + lane_id);
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        value_val_r.x[i] = (float)value_buf_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&value_cache[ite * offset] + lane_id) = value_val_r.v;
    }
    for (int i = 0; i < elems_per_thread; ++i)
    {
      sum_r[i] += (float)value_val_r.x[i] * logits[ite]; 
    }
  }
  for (int i = 0; i < elems_per_thread; i++)
  {
    sq[warp_id * WARP_SIZE + lane_id].x[i] = sum_r[i];
  }
  __syncthreads();
  if (warp_id == 0)
  {
    #pragma unroll
    for (int j = 1; j < warp_num; j++)
    {
      for (int i = 0; i < elems_per_thread; ++i)
      {
        sum_r[i] = sum_r[i] + (float)sq[j * WARP_SIZE + tid].x[i];
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < elems_per_thread; i++)
  {
    value_val_r.x[i] = sum_r[i];
  }
  if (warp_id == 0)
  {
    *((copy_t *)context_buf + lane_id) = value_val_r.v;
  }
}

template <typename T, int size_per_head, int block_sz>
__global__ 
void cross_attention_kernel_opt(
  T* __restrict query_buf, const T* __restrict Q_bias, 
  T* __restrict key_cache, const T* __restrict K_bias, 
  T* __restrict value_cache, const T* __restrict V_bias,
  const int* length_per_sample, T* __restrict context_buf, 
  int batch_size, int head_num, const int step, const int seq_len, const float scalar)
{  
  typedef Copy_t<T, size_per_head> copy_t;
  const int elems_per_thread = size_per_head / WARP_SIZE;
  union Access_t
  {
    copy_t v;
    T x[elems_per_thread]; // supported size 1,2,4
  };
  typedef struct Float_n_t
  {
    float x[elems_per_thread]; // supported size 1,2,4
  } float_n_t;

  __shared__ float_n_t sq[block_sz];
  __shared__ float logits[1024];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int warp_num = block_sz / WARP_SIZE;

  typedef cub::BlockReduce<float, block_sz> MaxValBlockReduce;
  typedef cub::BlockReduce<float, block_sz> BlockReduce;
  __shared__ typename MaxValBlockReduce::TempStorage max_val_block_temp_storage;
  __shared__ typename BlockReduce::TempStorage block_temp_storage;

  __shared__ typename cub::WarpReduce<float>::TempStorage temp_storage[warp_num];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x / head_num;
  const int head_id = blockIdx.x % head_num;

  int length = __ldg(&length_per_sample[bid]);

  const int lane_id = tid % WARP_SIZE;

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head;
  int qkv_bias_id = head_id * size_per_head;

  int key_value_id = bid * (seq_len * head_num * size_per_head) + 
  + head_id * size_per_head;

  query_buf = &query_buf[qkv_id];
  K_bias = &K_bias[qkv_bias_id];
  key_cache = &key_cache[key_value_id];
  Q_bias = &Q_bias[qkv_bias_id];
  V_bias = &V_bias[qkv_bias_id];
  value_cache = &value_cache[key_value_id];
  context_buf = &context_buf[qkv_id];

  Access_t bias_r, key_val_r, query_buf_r;

  // each warp will have its own copy of sq
  query_buf_r.v = *((copy_t *)query_buf + lane_id);
  bias_r.v = *((copy_t *)Q_bias + lane_id);
  float qb_r[elems_per_thread];
  for (int i = 0; i < elems_per_thread; ++i)
  {
    qb_r[i] =  (float)query_buf_r.x[i] + (float)bias_r.x[i];
  }

  //offset for each step
  int offset =  head_num * size_per_head;

  bias_r.v = *((copy_t *) K_bias + lane_id);
  for(int ite = warp_id; ite < length; ite += warp_num)
  {
    key_val_r.v = *((copy_t *)&key_cache[ite * offset] + lane_id);

    //For the first step, we should add bias to key memory cache.
    //The KV memory cache only need to be updated at the first step.
    if (step == 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        key_val_r.x[i] = (float)key_val_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&key_cache[ite * offset] + lane_id) = key_val_r.v;
    }
    float val = 0.f;
    for (int i = 0; i < elems_per_thread; i++)
    {
      val = val +  (float)key_val_r.x[i] * qb_r[i] * scalar;
    }
    float qk = cub::WarpReduce<float>(temp_storage[warp_id]).Sum(val);
    if (lane_id == 0)
    {
      logits[ite] = qk; 
    }
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;
  float local_i = -1e20f;
  for(int i = tid; i < length; i += blockDim.x)
    local_i = max(local_i, logits[i]);

  float max_val = MaxValBlockReduce(max_val_block_temp_storage).Reduce(local_i, cub::Max());
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  float local_o = 0.0f;
  for(int i = tid; i < length; i += blockDim.x)
  {
    logits[i] = __expf(logits[i] - s_max_val);
    local_o += logits[i];
  }
  float val = BlockReduce(block_temp_storage).Sum(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();

  float s_sum_inverse = __fdividef(1.0f, s_sum);
  for(int i = tid; i < length; i += blockDim.x)
  {
    logits[i] = logits[i] * s_sum_inverse;
  }
  __syncthreads(); 

  // This optimization introduces discrepancy because of different order in FP32 summation
  float sum_r[elems_per_thread] = {0.f};
  bias_r.v = *((copy_t *) V_bias + lane_id);
  for(int ite = warp_id; ite < length; ite += warp_num)
  {
    key_val_r.v = *((copy_t *)&value_cache[ite * offset] + lane_id);

    //For the first step, we should add bias to key memory cache.
    if(step == 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        key_val_r.x[i] = (float)key_val_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&value_cache[ite * offset] + lane_id) = key_val_r.v;
    }
    for (int i = 0; i < elems_per_thread; ++i)
    {
      sum_r[i] += (float)key_val_r.x[i] * logits[ite]; 
    }
  }
  for (int i = 0; i < elems_per_thread; i++)
  {
    sq[warp_id * WARP_SIZE + lane_id].x[i] = sum_r[i];
  }
  __syncthreads();
  if (threadIdx.x < WARP_SIZE)
  {
    #pragma unroll
    for (int j = 1; j < warp_num; j++)
    {
      for (int i = 0; i < elems_per_thread; ++i)
      {
        sum_r[i] = sum_r[i] + (float)sq[j * WARP_SIZE + threadIdx.x].x[i];
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < elems_per_thread; i++)
  {
    key_val_r.x[i] = sum_r[i];
  }
  if (threadIdx.x  < WARP_SIZE)
  {
    *((copy_t *)context_buf + lane_id) = key_val_r.v;
  }
}

template<typename T>
__global__
void cross_attention_kernel(
  T* query_buf, const T* Q_bias,
  T* key_cache, const T* K_bias,
  T* value_cache, const T* V_bias,
  const int* length_per_sample, T* context_buf, 
  int batch_size, int head_num, int size_per_head, int step, const int seq_len, const T scalar)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;

  extern __shared__ __align__(sizeof(T)) unsigned s_buf[];
  T* sq = reinterpret_cast<T *>(s_buf);
  T* logits = reinterpret_cast<T *>(&sq[size_per_head]);

  int length = __ldg(&length_per_sample[bid]);

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head + tid;
  int qkv_bias_id = head_id * size_per_head + tid;

  if(tid < size_per_head)
    sq[tid] = query_buf[qkv_id] + Q_bias[qkv_bias_id];
  __syncthreads();

  for(int ite = 0; ite < length; ++ite)
  {
    int key_id = bid * (seq_len * head_num * size_per_head) + ite * (head_num * size_per_head)
     + head_id * size_per_head + tid;

    T key = tid < size_per_head ? key_cache[key_id] : (T)(0.0f);

    //For the first step, we should add bias to key memory cache.
    //The KV memory cache only need to be updated at the first step.
    if(step == 1 && tid < size_per_head)
    {
      key += K_bias[head_id * size_per_head + tid];
      key_cache[key_id] = key;
    }

    T val = (tid < size_per_head) ? key * sq[tid] * scalar : (T)(0.0f);
    T qk = blockReduceSum(val);
    if(threadIdx.x == 0)
      logits[ite] = qk;
    __syncthreads(); //try to remove
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;

  float local_i = tid < length ? (float)logits[tid] : -1e20f; 
  float max_val = blockReduceMax<float>(local_i);
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  local_i -= s_max_val;
  float local_o = tid < length ? __expf(local_i) : 0.0f;
  float val = blockReduceSum<float>(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();
  if(tid < length)
    logits[tid] = local_o / s_sum;
  __syncthreads();

  if(tid < size_per_head)
  {
    T sum = (T)0.0f;
    for(int ite = 0; ite < length; ++ite)
    {
      int value_id = bid * seq_len * head_num * size_per_head + ite * head_num * size_per_head 
        + head_id * size_per_head + tid;

      T value = value_cache[value_id];

      //for the first step, we should add bias to key memory cache
      if(step == 1)
      {
        value += V_bias[head_id * size_per_head + tid];
        value_cache[value_id] = value;
      }  
      sum += value * logits[ite];
    }
    context_buf[bid * head_num * size_per_head + head_id * size_per_head + tid] = sum;
  }
}

#endif


template <typename T>
void cross_attention_dispatch(void* query_buf, const void* Q_bias, 
  void* key_cache, const void* K_bias, void* value_cache, const void* V_bias, const int* length,
  void* context_buf, int& batch_size, int& head_num, int& size_per_head, int& step, int seq_len, cudaStream_t stream)
  {
    const int block_sz = ATTENTION_BLOCK_SIZE;
    float scalar = 1.f / sqrtf(size_per_head * 1.0f);

    dim3 grid(batch_size * head_num);

    int cond = size_per_head * ((ATTENION_OPT)? 1:0);
    // printf("cond:%d\n",cond);
    switch (cond)
    {
      case 32:
        cross_attention_kernel_opt<T, 32, block_sz><<<grid, block_sz, 0, stream>>>(
          (T*)query_buf, (T*)Q_bias, (T*)key_cache,(T*)K_bias, (T*)value_cache, (T*)V_bias, length, (T*)context_buf,  
          batch_size, head_num, step, seq_len, scalar);
        break;
      case 64:
        cross_attention_kernel_opt<T, 64, block_sz><<<grid, block_sz, 0, stream>>>(
          (T*)query_buf, (T*)Q_bias, (T*)key_cache,(T*)K_bias, (T*)value_cache, (T*)V_bias, length, (T*)context_buf,  
          batch_size, head_num, step, seq_len, scalar);
        break;
      case 128:
        cross_attention_kernel_opt<T, 128, block_sz><<<grid, block_sz, 0, stream>>>(
          (T*)query_buf, (T*)Q_bias, (T*)key_cache,(T*)K_bias, (T*)value_cache, (T*)V_bias, length, (T*)context_buf,  
          batch_size, head_num, step, seq_len, scalar);
        break;
      default:
        // default path

        int block_size = 128;

        if(seq_len <= 64)
          block_size = 64;
        else if(seq_len <= 128 && seq_len > size_per_head)
          block_size = 128;
        else if(seq_len > 128 && seq_len <= 256)
          block_size = 256;
        else if(seq_len > 256 && seq_len <= 512)
          block_size = 512;
        else
          block_size = 1024;

        if(block_size < size_per_head)
          block_size = size_per_head;

        assert(block_size <= 1024);
        dim3 block(block_size);
        
        int shared_size = sizeof(T) * (size_per_head + seq_len);
        cross_attention_kernel<T><<<grid, block, shared_size, stream>>>(
          (T*)query_buf, (T*)Q_bias, 
          (T*)key_cache, (T*)K_bias,
          (T*)value_cache, (T*)V_bias,
          length, (T*)context_buf,  
          batch_size,
          head_num, size_per_head, step, seq_len, scalar);
    }
  }


template <typename T>
void masked_attention_dispatch(
  void* key_buf, void* value_buf,
  void* query_buf, const void* self_Q_bias, 
  void* key_cache, const void* self_K_bias, void* value_cache, const void* self_V_bias,
  void* context_buf, int& batch_size, int& head_num, int& size_per_head, const int& step, cudaStream_t stream, int* padding_index)
  {
    #if 1
    const int block_sz = ATTENTION_BLOCK_SIZE;
    T scalar = (T)(1.f / sqrtf(size_per_head * 1.0f));

    dim3 grid(batch_size * head_num);

    int cond = size_per_head * ((ATTENION_OPT)? 1:0);
    switch (cond)
    {
      case 32:
        masked_attention_kernel_opt<32, block_sz, T><<<grid, block_sz, 0, stream>>>(
          (T*)key_buf, (T*)value_buf,
          (T*)query_buf, (T*)self_Q_bias,  (T*)key_cache, (T*)self_K_bias, (T*)value_cache, (T*)self_V_bias, (T*)context_buf, 
          batch_size, head_num, step, scalar, padding_index); 
        break;
      case 64:
        if(sizeof(T) == 2)
          masked_attention_kernel_opt_half2<64, block_sz><<<grid, block_sz, 0, stream>>>(
            (T*)key_buf, (T*)value_buf,
            (T*)query_buf, (T*)self_Q_bias,  (T*)key_cache, (T*)self_K_bias, (T*)value_cache, (T*)self_V_bias, (T*)context_buf, 
            batch_size, head_num, step, scalar, padding_index);
        else
          masked_attention_kernel_opt<64, block_sz, T><<<grid, block_sz, 0, stream>>>(
            (T*)key_buf, (T*)value_buf,
            (T*)query_buf, (T*)self_Q_bias,  
            (T*)key_cache, (T*)self_K_bias, 
            (T*)value_cache, (T*)self_V_bias, 
            (T*)context_buf, 
            batch_size, head_num, step, scalar, padding_index);
        break;
      case 80:
        if(sizeof(T) == 2)
          masked_attention_kernel_opt_half2<80, block_sz><<<grid, block_sz, 0, stream>>>(
            (T*)key_buf, (T*)value_buf,
            (T*)query_buf, (T*)self_Q_bias,  (T*)key_cache, (T*)self_K_bias, (T*)value_cache, (T*)self_V_bias, (T*)context_buf, 
            batch_size, head_num, step, scalar, padding_index);
        else
          masked_attention_kernel_opt<80, block_sz, T><<<grid, block_sz, 0, stream>>>(
            (T*)key_buf, (T*)value_buf,
            (T*)query_buf, (T*)self_Q_bias,  
            (T*)key_cache, (T*)self_K_bias, 
            (T*)value_cache, (T*)self_V_bias, 
            (T*)context_buf, 
            batch_size, head_num, step, scalar, padding_index);
        break;
      case 96:
        if(sizeof(T) == 2)
          masked_attention_kernel_opt_half2<96, block_sz><<<grid, block_sz, 0, stream>>>(
            (T*)key_buf, (T*)value_buf,
            (T*)query_buf, (T*)self_Q_bias,  (T*)key_cache, (T*)self_K_bias, (T*)value_cache, (T*)self_V_bias, (T*)context_buf, 
            batch_size, head_num, step, scalar, padding_index);
        else
          masked_attention_kernel_opt<96, block_sz, T><<<grid, block_sz, 0, stream>>>(
            (T*)key_buf, (T*)value_buf,
            (T*)query_buf, (T*)self_Q_bias,  
            (T*)key_cache, (T*)self_K_bias, 
            (T*)value_cache, (T*)self_V_bias, 
            (T*)context_buf, 
            batch_size, head_num, step, scalar, padding_index);
        break;
      case 128:
        if(sizeof(T) == 2)
          masked_attention_kernel_opt_half2<128, block_sz><<<grid, block_sz, 0, stream>>>(
            (T*)key_buf, (T*)value_buf,
            (T*)query_buf, (T*)self_Q_bias,  (T*)key_cache, (T*)self_K_bias, (T*)value_cache, (T*)self_V_bias, (T*)context_buf, 
            batch_size, head_num, step, scalar, padding_index);
        else
          masked_attention_kernel_opt<128, block_sz, T><<<grid, block_sz, 0, stream>>>(
            (T*)key_buf, (T*)value_buf,
            (T*)query_buf, (T*)self_Q_bias,  (T*)key_cache, (T*)self_K_bias, (T*)value_cache, (T*)self_V_bias, (T*)context_buf, 
            batch_size, head_num, step, scalar, padding_index);
        break;
      default:
        // default path
        int block_size = 128;
        
        //suppose size_per_head <= 128
        if(step <= 64)
          block_size = 64;
        else if(step <= 128 && step > size_per_head)
          block_size = 128;
        else if(step > 128 && step <= 256)
          block_size = 256;
        else if(step > 256 && step <= 512)
          block_size = 512;
        else
          block_size = 1024;
        
        if((int)block_size < size_per_head)
          block_size = size_per_head;
          
        assert(block_size <= 1024);
        dim3 block(block_size);
        T scalar = 1 / sqrtf(size_per_head * 1.0f);

        
        int shared_size = sizeof(T) * (size_per_head + step);
        masked_attention_kernel<T><<<grid, block, shared_size, stream>>>(
          (T*)key_buf, (T*)value_buf,
          (T*)query_buf, (T*)self_Q_bias, 
          (T*)key_cache, (T*)self_K_bias,
          (T*)value_cache, (T*)self_V_bias,
          (T*)context_buf, batch_size,
          head_num, size_per_head, step, scalar, padding_index);
    }
   #endif
  }


template void masked_attention_dispatch<float>(void* key_buf, void* value_buf,
                                              void* query_buf, const void* self_Q_bias, 
                                              void* key_cache, const void* self_K_bias, void* value_cache, const void* self_V_bias,
                                              void* context_buf, int& batch_size, int& head_num, int& size_per_head, const int& step, cudaStream_t stream, int* padding_index);

template void masked_attention_dispatch<half>(void* key_buf, void* value_buf,
                                              void* query_buf, const void* self_Q_bias, 
                                              void* key_cache, const void* self_K_bias, void* value_cache, const void* self_V_bias,
                                              void* context_buf, int& batch_size, int& head_num, int& size_per_head, const int& step, cudaStream_t stream, int* padding_index);

template void cross_attention_dispatch<float>(void *query_buf, const void *Q_bias,
                                              void *key_cache, const void *K_bias, void *value_cache, const void *V_bias, const int *length,
                                              void *context_buf, int &batch_size, int &head_num, int &size_per_head, int &step, int seq_len, cudaStream_t stream);

template void cross_attention_dispatch<half>(void *query_buf, const void *Q_bias,
                                              void *key_cache, const void *K_bias, void *value_cache, const void *V_bias, const int *length,
                                              void *context_buf, int &batch_size, int &head_num, int &size_per_head, int &step, int seq_len, cudaStream_t stream);