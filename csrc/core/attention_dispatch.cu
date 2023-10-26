#include "core/common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <assert.h>
#include "cub/cub.cuh"
#include <iostream>
#include <vector>

const int WARP_SIZE = 32;
const bool ATTENION_OPT = true;
const int ATTENTION_BLOCK_SIZE = 256;

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


// attention_dispatch code modified from Nvidia's DeepLearningExamples
// https://github.com/NVIDIA/DeepLearningExamples/blob/master/FasterTransformer/v3.1/fastertransformer/cuda/open_decoder.cu#L194-L824

#if 1
template <typename T>
__global__ 
void masked_attention_kernel(
  T* key_buf, T* value_buf,
  T* query_buf, const T* self_Q_bias, 
  T* key_cache, const T* self_K_bias,
  T* value_cache, const T* self_V_bias,
  T* context_buf, int first_batch_size, int head_num,
  int size_per_head, const int step, const T scalar,
  const int64_t* pre_padding_len,const int64_t *reorder_index)
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

  int cache_id = batch_id;
  if (reorder_index != nullptr){
      cache_id = reorder_index[batch_id];
  }

  int inx_id = cache_id * head_num * size_per_head + head_id * size_per_head + tid;


  if(tid < size_per_head)
    sq[tid] = query_buf[qkv_id] + self_Q_bias[qkv_bias_id];
  __syncthreads();

  int padding_len = 0;
  if (pre_padding_len != nullptr){
      padding_len = pre_padding_len[batch_id];
  }

  //offset for each step
  int offset = first_batch_size * head_num * size_per_head;
  for(int ite = 0; ite < step; ++ite)
  { 
    T key = tid < size_per_head ? key_cache[ite * offset + inx_id] : (T)0.0f;
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1 && tid < size_per_head)
    {
      key = key_buf[qkv_id] + self_K_bias[qkv_bias_id];
      key_cache[ite * offset + inx_id] = key;
    }


    T val = (tid < size_per_head) ? key * sq[tid] * scalar : (T)(0.0f);

    if(ite >= padding_len)
    {
      T qk = blockReduceSum(val);
      if(threadIdx.x == 0){
        logits[ite] = qk;
      }
    }
    else
    {
      if(threadIdx.x == 0){
        logits[ite] = (T)-1e20f;
      }
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
      T value = value_cache[ite * offset + inx_id];
      //for the last step, we should update K + bias_K to the cache
      if(ite == step - 1)
      {
        value = value_buf[qkv_id] + self_V_bias[qkv_bias_id];
        value_cache[ite * offset + inx_id] = value;
      }
      sum += value * logits[ite];
    }
    context_buf[qkv_id] = sum;
  }
}

template <int size_per_head, int block_sz, typename T>
__global__ 
void masked_attention_kernel_opt(
  T* __restrict key_buf, T* __restrict value_buf, T* __restrict query_buf,
  T* __restrict key_cache, T* __restrict value_cache, T* __restrict context_buf,
  int first_batch_size, int head_num, const int step, const T scalar,
  const int64_t* pre_padding_len, const int64_t *reorder_index)
{
  // typedef Copy_t<T, size_per_head> copy_t;
  const int elems_per_thread = size_per_head / WARP_SIZE;

  typedef struct Float_n_t
  {
    T x[elems_per_thread];
  } float_n_t;

  union Access_t
  {
    float_n_t v;
    T x[elems_per_thread];
  };

  __shared__ float_n_t sq[block_sz];

  __shared__ float logits[4096]; // only use [0 ~ step-1], the step should be smaller than 4096

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
  int cache_id = batch_id;
  if (reorder_index != nullptr){
      cache_id = reorder_index[batch_id];
  }
  int inx_id = (cache_id * head_num + head_id) * size_per_head;
  // printf("bid:%d cache_id:%d batch_id:%d inx_id:%d qkv_id:%d\n",bid,cache_id,batch_id,inx_id,qkv_id);
  // int qkv_id = bid * size_per_head;

  query_buf = &query_buf[qkv_id];
  key_buf = &key_buf[qkv_id];
  value_buf = &value_buf[qkv_id];
  key_cache = &key_cache[inx_id];
  value_cache = &value_cache[inx_id];
  context_buf = &context_buf[qkv_id];

  Access_t bias_r, query_buf_r;
  Access_t key_val_r, key_buf_r;
  Access_t value_val_r, value_buf_r;

  // each warp will have its own copy of sq
  query_buf_r.v = *((float_n_t *)query_buf + lane_id);
  key_buf_r.v = *((float_n_t *)key_buf + lane_id);
  float qb_r[elems_per_thread];
  for (int i = 0; i < elems_per_thread; ++i)
  {
    qb_r[i] =  (float)query_buf_r.x[i];
  }
  int padding_len = 0;
  if (pre_padding_len != nullptr){
      padding_len = pre_padding_len[cache_id];
  }
  // printf("padding_len:%d cache_id:%d batch_id:%d\n",padding_len,cache_id,batch_id);

  //offset for each step
  int offset = first_batch_size * head_num * size_per_head;
  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    if (ite < padding_len){
        logits[ite] = (T)-1e20f;
    }else {
        key_val_r.v = *((float_n_t *) &key_cache[ite * offset] + lane_id);
        //for the last step, we should update K + bias_K to the cache
        if (ite == step - 1) {
            for (int i = 0; i < elems_per_thread; i++) {
                key_val_r.x[i] = (float) key_buf_r.x[i];
            }
            *((float_n_t *) &key_cache[ite * offset] + lane_id) = key_val_r.v;
            //*((copy_t *)&key_cache_t[ite * size_per_head] + lane_id) = key_val_r.v;
        }
        float val = 0.f;
        for (int i = 0; i < elems_per_thread; i++) {
            val = val + (float) key_val_r.x[i] * qb_r[i] * (float) scalar;
        }
        float qk = cub::WarpReduce<float>(temp_storage[warp_id]).Sum(val);
        if (lane_id == 0)
        {
            logits[ite] = (T)qk;
        }
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
  value_buf_r.v = *((float_n_t *)value_buf + lane_id);

  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    if(ite < padding_len)
        continue;
    value_val_r.v = *((float_n_t *)&value_cache[ite * offset] + lane_id);
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        value_val_r.x[i] = (float)value_buf_r.x[i];
      }
      *((float_n_t *)&value_cache[ite * offset] + lane_id) = value_val_r.v;
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
    *((float_n_t *)context_buf + lane_id) = value_val_r.v;
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

template <typename T, int size_per_head, int block_sz>
__global__
void t5_cross_attention_kernel_opt(
  T* __restrict query_buf,
  T* __restrict key_cache,
  T* __restrict value_cache,
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

  int key_value_id = bid * (seq_len * head_num * size_per_head) +
  + head_id * size_per_head;

  query_buf = &query_buf[qkv_id];
  key_cache = &key_cache[key_value_id];
  value_cache = &value_cache[key_value_id];
  context_buf = &context_buf[qkv_id];

  Access_t key_val_r, query_buf_r;

  // each warp will have its own copy of sq
  query_buf_r.v = *((copy_t *)query_buf + lane_id);
  float qb_r[elems_per_thread];
  for (int i = 0; i < elems_per_thread; ++i)
  {
    qb_r[i] =  (float)query_buf_r.x[i];
  }

  //offset for each step
  int offset =  head_num * size_per_head;

  for(int ite = warp_id; ite < length; ite += warp_num)
  {
    key_val_r.v = *((copy_t *)&key_cache[ite * offset] + lane_id);

    //For the first step, we should add bias to key memory cache.
    //The KV memory cache only need to be updated at the first step.
    if (step == 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        key_val_r.x[i] = (float)key_val_r.x[i];
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
  for(int ite = warp_id; ite < length; ite += warp_num)
  {
    key_val_r.v = *((copy_t *)&value_cache[ite * offset] + lane_id);

    //For the first step, we should add bias to key memory cache.
    if(step == 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        key_val_r.x[i] = (float)key_val_r.x[i];
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
    sq[tid] = Q_bias != nullptr ? query_buf[qkv_id] + Q_bias[qkv_bias_id] : query_buf[qkv_id];
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
      if (K_bias != nullptr)
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
        if (V_bias != nullptr)
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
  void* context_buf, int& batch_size, int& head_num, int& size_per_head, int& step, int &seq_len, cudaStream_t stream)
  {
    // printf("test cross attn fix********\n");
    const int block_sz = ATTENTION_BLOCK_SIZE;
    float scalar = Q_bias!=nullptr ? 1.f / sqrtf(size_per_head * 1.0f) : 1.0f;

    dim3 grid(batch_size * head_num);

    int cond = size_per_head * ((ATTENION_OPT)? 1:0);
    if (Q_bias != nullptr) {
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
        case 96:
          cross_attention_kernel_opt<T, 96, block_sz><<<grid, block_sz, 0, stream>>>(
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
    } else {
      switch (cond)
      {
        case 32:
          t5_cross_attention_kernel_opt<T, 32, block_sz><<<grid, block_sz, 0, stream>>>(
              (T *)query_buf, (T *)key_cache, (T *)value_cache, length, (T *)context_buf,
              batch_size, head_num, step, seq_len, scalar);
          break;
        case 64:
          t5_cross_attention_kernel_opt<T, 64, block_sz><<<grid, block_sz, 0, stream>>>(
              (T *)query_buf, (T *)key_cache, (T *)value_cache, length, (T *)context_buf,
              batch_size, head_num, step, seq_len, scalar);
          break;
        case 96:
          t5_cross_attention_kernel_opt<T, 96, block_sz><<<grid, block_sz, 0, stream>>>(
              (T *)query_buf, (T *)key_cache, (T *)value_cache, length, (T *)context_buf,
              batch_size, head_num, step, seq_len, scalar);
          break;
        case 128:
          t5_cross_attention_kernel_opt<T, 128, block_sz><<<grid, block_sz, 0, stream>>>(
              (T *)query_buf, (T *)key_cache, (T *)value_cache, length, (T *)context_buf,
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
  }


template <typename T>
void masked_attention_dispatch(void *key_buf, void *value_buf, void *query_buf, 
                               void *key_cache, void *value_cache, void *context_buf, 
                               int &batch_size, int &first_batch_size, int &head_num, 
                               int &size_per_head, const int &step, cudaStream_t stream, 
                               const int64_t *pre_padding_len, const int64_t *reorder_index)
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
          (T*)key_buf, (T*)value_buf, (T*)query_buf, (T*)key_cache, (T*)value_cache,
          (T*)context_buf, first_batch_size, head_num, step, scalar, pre_padding_len,reorder_index);
        break;
      case 64:
        masked_attention_kernel_opt<64, block_sz, T><<<grid, block_sz, 0, stream>>>(
          (T*)key_buf, (T*)value_buf, (T*)query_buf, (T*)key_cache, (T*)value_cache,
          (T*)context_buf, first_batch_size, head_num, step, scalar, pre_padding_len,reorder_index);
        break;
      case 96:
        masked_attention_kernel_opt<96, block_sz, T><<<grid, block_sz, 0, stream>>>(
          (T*)key_buf, (T*)value_buf, (T*)query_buf, (T*)key_cache, (T*)value_cache,
          (T*)context_buf, first_batch_size, head_num, step, scalar, pre_padding_len,reorder_index);
        break;
      case 128:
        masked_attention_kernel_opt<128, block_sz, T><<<grid, block_sz, 0, stream>>>(
          (T*)key_buf, (T*)value_buf, (T*)query_buf, (T*)key_cache, (T*)value_cache,
          (T*)context_buf, first_batch_size, head_num, step, scalar, pre_padding_len,reorder_index);
        break;
      default:
        std::cerr << "not support size_per_head: " << size_per_head << " for attention" << std::endl;
    }
   #endif
  }


template void masked_attention_dispatch<float>(void *key_buf, void *value_buf, void *query_buf,
                                               void *key_cache, void *value_cache, void *context_buf,
                                               int &batch_size, int &first_batch_size, int &head_num,
                                               int &size_per_head, const int &step, cudaStream_t stream,
                                               const int64_t *pre_padding_len, const int64_t *reorder_index);

template void masked_attention_dispatch<half>(void *key_buf, void *value_buf, void *query_buf,
                                               void *key_cache, void *value_cache, void *context_buf,
                                               int &batch_size, int &first_batch_size, int &head_num,
                                               int &size_per_head, const int &step, cudaStream_t stream,
                                               const int64_t *pre_padding_len, const int64_t *reorder_index);

template void cross_attention_dispatch<float>(void *query_buf, const void *Q_bias,
                                              void *key_cache, const void *K_bias, void *value_cache, const void *V_bias, const int *length,
                                              void *context_buf, int &batch_size, int &head_num, int &size_per_head, int &step, int &seq_len, cudaStream_t stream);

template void cross_attention_dispatch<half>(void *query_buf, const void *Q_bias,
                                              void *key_cache, const void *K_bias, void *value_cache, const void *V_bias, const int *length,
                                              void *context_buf, int &batch_size, int &head_num, int &size_per_head, int &step, int &seq_len, cudaStream_t stream);





template <int size_per_head, int block_sz, typename T>
__global__ 
void fused_masked_attention_kernel(
  T* __restrict qkv_buf, const T* __restrict self_Q_bias, 
  T* __restrict key_cache, const T* __restrict self_K_bias, 
  T* __restrict value_cache, const T* __restrict self_V_bias,
  T* __restrict context_buf, int first_batch_size, int head_num, 
  const int step, const T scalar, const int64_t* pre_padding_len,const int64_t *reorder_index)
{
  // typedef Copy_t<T, size_per_head> copy_t;
  const int elems_per_thread = size_per_head / WARP_SIZE;

  typedef struct Float_n_t
  {
    T x[elems_per_thread];
  } float_n_t;

  union Access_t
  {
    float_n_t v;
    T x[elems_per_thread];
  };

  __shared__ float_n_t sq[block_sz];

  __shared__ float logits[4096]; // only use [0 ~ step-1], the step should be smaller than 4096

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
  int cache_id = batch_id;
  if (reorder_index != nullptr){
      cache_id = reorder_index[batch_id];
  }
  int inx_id = (cache_id * head_num + head_id) * size_per_head;
  // printf("bid:%d batch_id:%d qkv_id:%d\n",bid,batch_id,qkv_id);
  // int qkv_id = bid * size_per_head;
  int q_id = qkv_id + 2 * size_per_head * head_num * batch_id;
  // printf("q_id:%d qkv_id:%d\n",q_id,qkv_id);


  qkv_buf = &qkv_buf[q_id];
  // key_buf = &qkv_buf[qkv_id];
  // value_buf = &qkv_buf[qkv_id];
  self_K_bias = &self_K_bias[qkv_bias_id];
  key_cache = &key_cache[inx_id];
  self_Q_bias = &self_Q_bias[qkv_bias_id];
  self_V_bias = &self_V_bias[qkv_bias_id];
  value_cache = &value_cache[inx_id];
  context_buf = &context_buf[qkv_id];

  Access_t bias_r, query_buf_r;
  Access_t key_val_r, key_buf_r;
  Access_t value_val_r, value_buf_r;

  // each warp will have its own copy of sq
  query_buf_r.v = *((float_n_t *)qkv_buf + lane_id);
  key_buf_r.v = *((float_n_t *)qkv_buf + lane_id + size_per_head * head_num/elems_per_thread);
  bias_r.v = *((float_n_t *)self_Q_bias + lane_id);
  float qb_r[elems_per_thread];
  for (int i = 0; i < elems_per_thread; ++i)
  {
    qb_r[i] =  (float)query_buf_r.x[i] + (float)bias_r.x[i];
    // printf("qb_r:%f %f %f %d\n",qb_r[i],(float)query_buf_r.x[i],(float)bias_r.x[i],q_id);
  }
  int padding_len = 0;
  if (pre_padding_len != nullptr){
      padding_len = pre_padding_len[cache_id];
  }
  // printf("padding_len:%d cache_id:%d batch_id:%d\n",padding_len,cache_id,batch_id);

  //offset for each step
  int offset = first_batch_size * head_num * size_per_head;
  bias_r.v = *((float_n_t *) self_K_bias + lane_id);
  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    if (ite < padding_len){
        logits[ite] = (T)-1e20f;
    }else {
        key_val_r.v = *((float_n_t *) &key_cache[ite * offset] + lane_id);
        //for the last step, we should update K + bias_K to the cache
        if (ite == step - 1) {
            for (int i = 0; i < elems_per_thread; i++) {
                key_val_r.x[i] = (float) key_buf_r.x[i] + (float) bias_r.x[i];
                // printf("%f %f %d %d\n",(float)key_buf_r.x[i],(float)bias_r.x[i],q_id, size_per_head * head_num);

            }
            *((float_n_t *) &key_cache[ite * offset] + lane_id) = key_val_r.v;
            //*((copy_t *)&key_cache_t[ite * size_per_head] + lane_id) = key_val_r.v;
        }
        float val = 0.f;
        for (int i = 0; i < elems_per_thread; i++) {
            val = val + (float) key_val_r.x[i] * qb_r[i] * (float) scalar;
        }
        float qk = cub::WarpReduce<float>(temp_storage[warp_id]).Sum(val);
        if (lane_id == 0)
        {
            logits[ite] = (T)qk;
        }
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
  bias_r.v = *((float_n_t *) self_V_bias + lane_id);
  value_buf_r.v = *((float_n_t *)qkv_buf + lane_id + 2 * size_per_head * head_num/elems_per_thread);

  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    if(ite < padding_len)
        continue;
    value_val_r.v = *((float_n_t *)&value_cache[ite * offset] + lane_id);
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        value_val_r.x[i] = (float)value_buf_r.x[i] + (float)bias_r.x[i];
      }
      *((float_n_t *)&value_cache[ite * offset] + lane_id) = value_val_r.v;
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
    *((float_n_t *)context_buf + lane_id) = value_val_r.v;
  }
}

template <int size_per_head, int block_sz, typename T>
__global__ 
void t5_masked_attention_kernel(
  T* __restrict qkv_buf, T* __restrict key_cache, T* __restrict value_cache, const T* __restrict position_bias,
  T* __restrict context_buf, int first_batch_size, int head_num, const int step, const T scalar, 
  const int64_t* pre_padding_len, const int64_t *reorder_index)
{
  // typedef Copy_t<T, size_per_head> copy_t;
  const int elems_per_thread = size_per_head / WARP_SIZE;

  typedef struct Float_n_t
  {
    T x[elems_per_thread];
  } float_n_t;

  union Access_t
  {
    float_n_t v;
    T x[elems_per_thread];
  };

  __shared__ float_n_t sq[block_sz];

  __shared__ float logits[4096]; // only use [0 ~ step-1], the step should be smaller than 4096

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
  int cache_id = batch_id;
  if (reorder_index != nullptr){
      cache_id = reorder_index[batch_id];
  }
  int inx_id = (cache_id * head_num + head_id) * size_per_head;
  int q_id = qkv_id + 2 * size_per_head * head_num * batch_id;


  qkv_buf = &qkv_buf[q_id];
  key_cache = &key_cache[inx_id];
  value_cache = &value_cache[inx_id];
  context_buf = &context_buf[qkv_id];

  Access_t query_buf_r;
  Access_t key_val_r, key_buf_r;
  Access_t value_val_r, value_buf_r;

  // each warp will have its own copy of sq
  query_buf_r.v = *((float_n_t *)qkv_buf + lane_id);
  key_buf_r.v = *((float_n_t *)qkv_buf + lane_id + size_per_head * head_num/elems_per_thread);
  float qb_r[elems_per_thread];
  for (int i = 0; i < elems_per_thread; ++i)
  {
    qb_r[i] =  (float)query_buf_r.x[i];
  }
  int padding_len = 0;
  if (pre_padding_len != nullptr){
      padding_len = pre_padding_len[cache_id];
  }

  //offset for each step
  int offset = first_batch_size * head_num * size_per_head;
  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    if (ite < padding_len){
        logits[ite] = (T)-1e20f;
    }else {
        key_val_r.v = *((float_n_t *) &key_cache[ite * offset] + lane_id);
        //for the last step, we should update K + bias_K to the cache
        if (ite == step - 1) {
            for (int i = 0; i < elems_per_thread; i++) {
                key_val_r.x[i] = (float)key_buf_r.x[i];
                // printf("%f %f %d %d\n",(float)key_buf_r.x[i],(float)bias_r.x[i],q_id, size_per_head * head_num);
            }
            *((float_n_t *) &key_cache[ite * offset] + lane_id) = key_val_r.v;
            //*((copy_t *)&key_cache_t[ite * size_per_head] + lane_id) = key_val_r.v;
        }
        float val = 0.f;
        for (int i = 0; i < elems_per_thread; i++) {
            val = val + (float) key_val_r.x[i] * qb_r[i] * (float) scalar;
        }
        float qk = cub::WarpReduce<float>(temp_storage[warp_id]).Sum(val);
        if (lane_id == 0)
        {
            logits[ite] = (position_bias!=nullptr) ? (T)qk + position_bias[head_id * step + ite] : (T)qk;
        }
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
  value_buf_r.v = *((float_n_t *)qkv_buf + lane_id + 2 * size_per_head * head_num/elems_per_thread);

  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    if(ite < padding_len)
        continue;
    value_val_r.v = *((float_n_t *)&value_cache[ite * offset] + lane_id);
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        value_val_r.x[i] = (float)value_buf_r.x[i];
      }
      *((float_n_t *)&value_cache[ite * offset] + lane_id) = value_val_r.v;
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
    *((float_n_t *)context_buf + lane_id) = value_val_r.v;
  }
}

template <typename T>
void fused_masked_attention_dispatch(
  void* qkv_buf,const void* self_Q_bias, 
  void* key_cache, const void* self_K_bias, 
  void* value_cache, const void* self_V_bias,
  void* context_buf, int& batch_size, int& first_batch_size,
  int& head_num, int& size_per_head,
  const int& step, cudaStream_t stream, 
  const int64_t* pre_padding_len, const int64_t* reorder_index, const void* position_bias)
  {
    const int block_sz = ATTENTION_BLOCK_SIZE;

    dim3 grid(batch_size * head_num);

    int cond = size_per_head * ((ATTENION_OPT)? 1:0);

    if (self_Q_bias != nullptr) {
      T scalar = (T)(1.f / sqrtf(size_per_head * 1.0f));
      switch (cond)
      {
        case 32:
          fused_masked_attention_kernel<32, block_sz, T><<<grid, block_sz, 0, stream>>>(
            (T*)qkv_buf,  (T*)self_Q_bias,  (T*)key_cache, (T*)self_K_bias, (T*)value_cache, (T*)self_V_bias, (T*)context_buf, 
            first_batch_size, head_num, step, scalar, pre_padding_len,reorder_index); 
          break;
        case 64:
          fused_masked_attention_kernel<64, block_sz, T><<<grid, block_sz, 0, stream>>>(
            (T*)qkv_buf, (T*)self_Q_bias,  
            (T*)key_cache, (T*)self_K_bias, 
            (T*)value_cache, (T*)self_V_bias, 
            (T*)context_buf, 
            first_batch_size, head_num, step, scalar, pre_padding_len,reorder_index);
          break;
        // case 80:
        //   masked_attention_kernel_opt<80, block_sz, T><<<grid, block_sz, 0, stream>>>(
        //     (T*)key_buf, (T*)value_buf,
        //     (T*)query_buf, (T*)self_Q_bias,  
        //     (T*)key_cache, (T*)self_K_bias, 
        //     (T*)value_cache, (T*)self_V_bias, 
        //     (T*)context_buf, 
        //     first_batch_size, head_num, step, scalar, pre_padding_len);
        //   break;
        case 96:
          fused_masked_attention_kernel<96, block_sz, T><<<grid, block_sz, 0, stream>>>(
            (T*)qkv_buf,(T*)self_Q_bias,  
            (T*)key_cache, (T*)self_K_bias, 
            (T*)value_cache, (T*)self_V_bias, 
            (T*)context_buf, 
            first_batch_size, head_num, step, scalar, pre_padding_len,reorder_index);
          break;
        case 128:
          fused_masked_attention_kernel<128, block_sz, T><<<grid, block_sz, 0, stream>>>(
            (T*)qkv_buf,(T*)self_Q_bias,  (T*)key_cache, (T*)self_K_bias, (T*)value_cache, (T*)self_V_bias, (T*)context_buf, 
            first_batch_size, head_num, step, scalar, pre_padding_len,reorder_index);
          break;
        default:
          assert(false);
      }
    } else {
      // T scalar = (T)(1.0f); // TODO T5 is different
      T scalar = (T)(1.f / sqrtf(size_per_head * 1.0f));
      switch (cond)
      {
        case 32:
          t5_masked_attention_kernel<32, block_sz, T><<<grid, block_sz, 0, stream>>>(
            (T*)qkv_buf, (T*)key_cache, (T*)value_cache, (T*)position_bias, (T*)context_buf, 
            first_batch_size, head_num, step, scalar, pre_padding_len, reorder_index); 
          break;
        case 64:
          t5_masked_attention_kernel<64, block_sz, T><<<grid, block_sz, 0, stream>>>(
            (T*)qkv_buf, (T*)key_cache, (T*)value_cache, (T*)position_bias, (T*)context_buf, 
            first_batch_size, head_num, step, scalar, pre_padding_len, reorder_index); 
          break;
        case 96:
          t5_masked_attention_kernel<96, block_sz, T><<<grid, block_sz, 0, stream>>>(
            (T*)qkv_buf, (T*)key_cache, (T*)value_cache, (T*)position_bias, (T*)context_buf, 
            first_batch_size, head_num, step, scalar, pre_padding_len, reorder_index); 
          break;
        case 128:
          t5_masked_attention_kernel<128, block_sz, T><<<grid, block_sz, 0, stream>>>(
            (T*)qkv_buf, (T*)key_cache, (T*)value_cache, (T*)position_bias, (T*)context_buf, 
            first_batch_size, head_num, step, scalar, pre_padding_len, reorder_index); 
          break;
        default:
          assert(false);
      }     
    }
  }


template void fused_masked_attention_dispatch<float>(void* qkv_buf,const void* self_Q_bias, 
                                              void* key_cache, const void* self_K_bias, void* value_cache, const void* self_V_bias,
                                              void* context_buf, int& batch_size,int& first_batch_size, int& head_num, int& size_per_head, const int& step, cudaStream_t stream, const int64_t* pre_padding_len,const int64_t *reorder_index, const void* position_bias);

template void fused_masked_attention_dispatch<half>(void* qkv_buf ,const void* self_Q_bias, 
                                              void* key_cache, const void* self_K_bias, void* value_cache, const void* self_V_bias,
                                              void* context_buf, int& batch_size,int& first_batch_size, int& head_num, int& size_per_head, const int& step, cudaStream_t stream, const int64_t* pre_padding_len,const int64_t *reorder_index, const void* position_bias);
