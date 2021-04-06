#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "core/common.cuh"
#include <assert.h>

// transpose kernel code modified from Nvidia's DeepLearningExamples
// https://github.com/NVIDIA/DeepLearningExamples/blob/master/FasterTransformer/v3.1/fastertransformer/cuda/open_attention.cu#L2156-L2182


template<typename T>
__global__
void transpose(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
  int batch_id = blockIdx.x / (head_num * seq_len);
  int seq_id = blockIdx.x % seq_len;
  int head_id = (blockIdx.x % (head_num * seq_len))/ seq_len;
  dst[batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head
    + head_id * size_per_head + threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

template<>
  __global__
void transpose(half* src, half* dst,
    const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int batch_id = tid / (head_num * seq_len * size_per_head);
  int head_id = (tid % (head_num * seq_len * size_per_head)) / (seq_len * size_per_head);
  int seq_id = (tid % (seq_len * size_per_head)) / size_per_head;
  int id = tid % size_per_head;

  int target_id = target_index(batch_id, head_id, seq_id, id, batch_size, head_num, seq_len, size_per_head);
  half2* src_ptr = (half2*)src;
  half2* dst_ptr = (half2*)dst;

  dst_ptr[target_id] = src_ptr[tid];
}

template<typename  T>
__global__
  // dim3 grid(batch_size * head_num * seq_len);
  // dim3 block(size_per_head);
void copyKV_transpose(T* __restrict src_k, T* __restrict src_v, T*  __restrict dest_k, T*  __restrict dest_v,
                    int size_per_head, int from_seq_len, int batch, int head_num){
    int bid = blockIdx.x / from_seq_len;
    int seq_id = blockIdx.x % from_seq_len;

    int src_idx = bid * from_seq_len * size_per_head + seq_id * size_per_head + threadIdx.x;
    int dest_idx = seq_id * batch * head_num  * size_per_head + bid * size_per_head + threadIdx.x;

    dest_k[dest_idx] = src_k[src_idx];
    dest_v[dest_idx] = src_v[src_idx];
}

  template<typename  T>
  __global__
  void copyKV_transpose_cross(T* __restrict src_k, T* __restrict src_v, T*  __restrict dest_k, T*  __restrict dest_v,
                      int size_per_head, int from_seq_len, int batch, int head_num){
      int bid = blockIdx.x / from_seq_len;
      int bid2 = blockIdx.x / batch;
      int seq_id = blockIdx.x % from_seq_len;
      int seq_id2 = blockIdx.x % batch;


      int src_idx = bid * from_seq_len * size_per_head + seq_id * size_per_head + threadIdx.x;
      int dest_idx = seq_id * batch * head_num  * size_per_head + bid * size_per_head + threadIdx.x;
      int dest_idx1 ;
      int embedding = head_num * size_per_head;

      if (dest_idx/(embedding*batch) >= 1)
      {
          dest_idx1 = (dest_idx/embedding)*(embedding*from_seq_len) + dest_idx%embedding - (dest_idx/(embedding*batch))*(embedding*from_seq_len*batch - embedding);   
      }
      else
      {
          dest_idx1 = (dest_idx/embedding)*(embedding*from_seq_len) + dest_idx%embedding;   
      }

      
      dest_k[dest_idx1] = src_k[src_idx];
      dest_v[dest_idx1] = src_v[src_idx];
  }

template<typename T>
__global__
void transpose_input(T* from_buf, T* input_buf, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int src_idx = blockIdx.x * size_per_head * head_num + threadIdx.x;
    int dest_idx = threadIdx.x * batch_size * seq_len + blockIdx.x;

    from_buf[dest_idx] = input_buf[src_idx];
}

template <typename T>
void transpose_kernel(void* transpose_dst, void* dst, const int& batch_size, const int& seq_len,
                const int& head_num, const int& size_per_head, const cudaStream_t stream){
            dim3 grid, block;
            if (sizeof(T) == sizeof(half)){
                const int seq_per_block = 4;
                int size_per_head_half = size_per_head / 2;
                grid.x = batch_size * head_num * seq_len / seq_per_block;
                block.x = seq_per_block * size_per_head_half;
                assert(grid.x * seq_per_block == batch_size * head_num * seq_len);
                transpose<T><<<grid, block, 0, stream>>>((T*)transpose_dst, (T*)dst, 
                      batch_size, seq_len, head_num, size_per_head_half);
            }else{
                const int seq_per_block = 1;
                grid.x = batch_size * head_num * seq_len / seq_per_block;
                block.x = seq_per_block * size_per_head;
                transpose<T><<<grid, block, 0, stream>>>((T*)transpose_dst,(T*)dst, 
                    batch_size, seq_len, head_num, size_per_head);
            }
        }

template <typename T>
void copyKV_transpose_kernel(void* d_K_buf, void* d_V_buf,void* K_buf, void* V_buf,const int& batch_size, 
                  const int& seq_len, const int& head_num, const int& size_per_head)
{
  dim3 grid(batch_size * head_num * seq_len);
  dim3 block(size_per_head);
  copyKV_transpose<T><<<grid, block>>>((T*)K_buf, (T*)V_buf, (T*)d_K_buf, (T*)d_V_buf, size_per_head,
                                            seq_len, batch_size, head_num);
}

template <typename  T>
void copyKV_transpose_cross_kernel(void* d_K_buf, void* d_V_buf,void* K_buf, 
                                      void* V_buf,const int& batch_size, const int& mem_seq_len,
                                      const int& head_num, const int& size_per_head) {
    dim3 grid(batch_size * head_num * mem_seq_len);
    dim3 block(size_per_head );
    copyKV_transpose_cross<T><<<grid, block>>>((T*)K_buf, (T*)V_buf, (T*)d_K_buf, (T*)d_V_buf, size_per_head,
            mem_seq_len, batch_size, head_num);
}


template void transpose_kernel<float>(void* transpose_dst, void* dst, const int& batch_size, const int& seq_len,
                                      const int& head_num, const int& size_per_head, const cudaStream_t stream);
template void transpose_kernel<half>(void* transpose_dst, void* dst, const int& batch_size, const int& seq_len,
                                      const int& head_num, const int& size_per_head, const cudaStream_t stream);

template void copyKV_transpose_kernel<float>(void* d_K_buf, void* d_V_buf,void* K_buf, void* V_buf,const int& batch_size, const int& seq_len,
                                      const int& head_num, const int& size_per_head);
template void copyKV_transpose_kernel<half>(void* d_K_buf, void* d_V_buf,void* K_buf, void* V_buf,const int& batch_size, const int& seq_len,
                                      const int& head_num, const int& size_per_head);

template void copyKV_transpose_cross_kernel<float>(void* d_K_buf, void* d_V_buf,void* K_buf, void* V_buf,const int& batch_size, const int& seq_len,
                                      const int& head_num, const int& size_per_head);
template void copyKV_transpose_cross_kernel<half>(void* d_K_buf, void* d_V_buf,void* K_buf, void* V_buf,const int& batch_size, const int& seq_len,
                                      const int& head_num, const int& size_per_head);
