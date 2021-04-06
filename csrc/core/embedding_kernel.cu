#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "core/common.cuh"
#include <assert.h>

template <typename T>
__global__ void embedding_lookup_acc(const T *embedding_table,const int hidden_units,const int64_t *input_ids,T *embedding_lookup_tensor,const int ratio ,const int num_id)
{
  int64_t ids = input_ids[blockIdx.x];
  int hidden_idx = threadIdx.x + blockIdx.y * blockDim.x;
  int hidden_size = hidden_units;
  T val = __ldg(&embedding_table[ids * hidden_size + hidden_idx]);

  for(int i = 0;i< ratio;i++)
  {
    embedding_lookup_tensor[i * num_id * hidden_size + blockIdx.x * hidden_size + hidden_idx] += val;
  }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
}


template <typename T>
__global__ void embedding_lookup(const T *embedding_table,const int hidden_units, const int64_t *input_ids, T *embedding_lookup_tensor,bool no_scale_embedding)
{
  int64_t ids = input_ids[blockIdx.x];
  int hidden_idx = threadIdx.x + blockIdx.y * blockDim.x;
  int hidden_size = hidden_units;
  T scale = 1.0;

  if( !no_scale_embedding)
  {
    scale = (T)sqrtf(float(hidden_units));
  }
  T val = __ldg(&embedding_table[ids * hidden_size + hidden_idx]) * scale;
  embedding_lookup_tensor[blockIdx.x * hidden_size + hidden_idx] = val ;
}

template <typename T>
void embedding_lookup_kernel(const void* embedding_table, const int64_t* input_ids, void* embedding_lookup_tensor,
  const int num_id,const int hidden_units, cudaStream_t stream,bool ifacc,const int ratio,bool no_scale_embedding)
{

  const int m = num_id;
  int k = hidden_units;
  assert(m <= 65536);
  int fold_coeff = 1;
  if (k <= 1024){
    fold_coeff = 1;
  }else if( k <= 2048){
    fold_coeff = 2;
  }else if(k <= 4096){
    fold_coeff = 4;
  }else if(k <= 8192){
    fold_coeff = 8;
  }else if(k <= 16384){
    fold_coeff = 16;
  }
    
  dim3 grid(m , fold_coeff);
  dim3 block(k / fold_coeff);

  // dim3 grid(num_id);
  // dim3 block(hidden_units);
  // assert(hidden_units <= 1024);
  if(ifacc)
  {
    embedding_lookup_acc<<<grid, block, 0, stream>>>((T*)embedding_table,hidden_units, input_ids, (T*)embedding_lookup_tensor,ratio,num_id);
  }
  else
  {
    embedding_lookup<<<grid, block, 0, stream>>>((T*)embedding_table, hidden_units,input_ids, (T*)embedding_lookup_tensor,no_scale_embedding);
  }
}

template void embedding_lookup_kernel<float>(const void* embedding_table, const int64_t* input_ids, void* embedding_lookup_tensor,
  const int num_id, const int hidden_units, cudaStream_t stream,bool ifacc,const int ratio,bool no_scale_embedding);

template void embedding_lookup_kernel<half>(const void* embedding_table, const int64_t* input_ids, void* embedding_lookup_tensor,
  const int num_id, const int hidden_units, cudaStream_t stream,bool ifacc,const int ratio,bool no_scale_embedding);




template<typename T>
__global__
void position_encoding(T* output, int seq_len, int padding_idx,int n){
  int tid = threadIdx.x + blockIdx.y * blockDim.x;
  int bid = blockIdx.x;
  float half_n = (float)n / 2.;
  
  int step = bid % seq_len + padding_idx + 1;
  float log_result = __logf(10000) / (half_n - 1.f);
  float exp_result = __expf( (tid % (int)half_n) * -1 * log_result );
  float scaled_time = exp_result *  step;
  
  T encoding_val = (tid < half_n) ? (T) __sinf(scaled_time) : (T) __cosf(scaled_time);
  output[bid * n + tid] = output[bid * n + tid]  + encoding_val;

}

template<typename T>
void position_encoding_kernel(
  void* output,
  int seq_len,int padding_idx,
  int m, int n, cudaStream_t stream)
{
  // dim3 grid(m);
  // dim3 block(n);
  // assert(n <= 1024);
  int fold_coeff = 1;
  int k = n;
  if (k <= 1024){
    fold_coeff = 1;
  }else if( k <= 2048){
    fold_coeff = 2;
  }else if(k <= 4096){
    fold_coeff = 4;
  }else if(k <= 8192){
    fold_coeff = 8;
  }else if(k <= 16384){
    fold_coeff = 16;
  }
  dim3 grid(m , fold_coeff);
  dim3 block(k / fold_coeff);
  position_encoding<T><<<grid, block, 0, stream>>>((T*)output, seq_len,padding_idx, n);
}

template void position_encoding_kernel<float>(void* output, int seq_len, int padding_idx,int m, int n, cudaStream_t stream);

template void position_encoding_kernel<half>(void* output, int seq_len, int padding_idx,int m, int n, cudaStream_t stream);