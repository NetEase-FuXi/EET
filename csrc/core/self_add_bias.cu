#include "core/self_add_bias.cuh"
#include <assert.h>
#include "core/common.cuh"

// add_QKV_bias kernel code modified from Nvidia's DeepLearningExamples
// https://github.com/NVIDIA/DeepLearningExamples/blob/master/FasterTransformer/v3.1/fastertransformer/cuda/open_attention.cu#L1342-L1395

template<typename T>
__global__
void add_QKV_bias_opt(T* Q, const T* bias_Q, T* K, const T* bias_K, T* V, const T* bias_V, T* q_buf_, T* k_buf_, T* v_buf_,
                  const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    T* data_ptr;
    T* buf_ptr;
    const T* bias_ptr;

    int m = batch_size * seq_len;
    int n = head_num * size_per_head;

    int qkv_id = blockIdx.x  /  m ;
    int row_offset = (blockIdx.x % m)  * n;

    if(qkv_id == 0)
    {
        data_ptr = Q + row_offset;
        buf_ptr = q_buf_;
        bias_ptr = bias_Q;
    }
    else if(qkv_id == 1)
    {
        data_ptr = K + row_offset;
        buf_ptr = k_buf_;
        bias_ptr = bias_K;
    }
    else
    {
        data_ptr = V + row_offset;
        buf_ptr = v_buf_;
        bias_ptr = bias_V;
    }

    int batch_id = (blockIdx.x  % m) / seq_len;
    int head_id = (threadIdx.x + blockIdx.y * blockDim.x) / size_per_head;
    int id_in_head = threadIdx.x % size_per_head;
    int word_start_id = (blockIdx.x ) % seq_len;

    T bias = __ldg(&bias_ptr[threadIdx.x + blockDim.x * blockIdx.y]);

    for(int i = word_start_id; i < word_start_id + 1; ++i)
    {
        T tmp = data_ptr[threadIdx.x + blockDim.x * blockIdx.y] + bias;

        int target_id = batch_id * (seq_len * head_num * size_per_head) + head_id * seq_len * size_per_head +
                        i * size_per_head + id_in_head;

        buf_ptr[target_id] = tmp;
        data_ptr += n;
    }
}


template<>
__global__
void add_QKV_bias_opt<half>( half* Q, const half* bias_Q, half* K, const half* bias_K, half* V, const half* bias_V,
                  half* q_buf_, half* k_buf_, half* v_buf_,
                  const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int tid = blockIdx.x * (size_per_head * head_num) + threadIdx.x + blockDim.x * blockIdx.y;
    int batch_id = tid / (head_num * seq_len * size_per_head);
    int seq_id = (tid % (head_num * seq_len * size_per_head)) / (head_num * size_per_head);
    int head_id = (tid % (head_num * size_per_head)) / size_per_head;
    int id = tid % size_per_head;
    int target_id = target_index(batch_id, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);

    int bias_id = threadIdx.x + blockDim.x * blockIdx.y;

    half2* src_ptr = (half2*)Q;
    half2* dst_ptr = (half2*)q_buf_;
    const half2* bias_ptr = (const half2*)bias_Q;
    dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));

    src_ptr = (half2*)K;
    dst_ptr = (half2*)k_buf_;
    bias_ptr = (const half2*)bias_K;
    dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));

    src_ptr = (half2*)V;
    dst_ptr = (half2*)v_buf_;
    bias_ptr = (const half2*)bias_V;
    dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));
}

template<typename T>
void add_QKV_bias_opt_kernel( void* Q, const void* bias_Q, void* K, const void* bias_K,  void* V, const void* bias_V, void* q_buf_, void* k_buf_, void* v_buf_,
                  const int& batch_size, const int& seq_len, const int& head_num, const int& size_per_head, const cudaStream_t stream){
        int qkv_types = 3;
        int m = batch_size * seq_len;
        int k = head_num * size_per_head;
        //assert(m * qkv_types <= 65536 && "batch_size * seq_len must <= 65536");
        int fold_coeff = 1;
        dim3 grid;
        dim3 block;
        //TODO - int8
        if (sizeof(T) == sizeof(float)){
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
            grid.x = m * qkv_types;
            grid.y = fold_coeff;
            block.x = k / fold_coeff;
            add_QKV_bias_opt<<<grid, block, 0, stream>>>((float*)Q, (float*)bias_Q, (float*)K, (float*)bias_K, (float*)V, (float*)bias_V, (float*)q_buf_, (float*)k_buf_, 
                (float*)v_buf_, batch_size, seq_len, head_num, size_per_head);
        }else{
            if (k <= 2048){
                fold_coeff = 2;
            }else if( k <= 4096){
                fold_coeff = 2;
            }else if(k <= 8192){
                fold_coeff = 4;
            }else if(k <= 16384){
                fold_coeff = 8;
            }else if(k <= 16384 * 2){
                fold_coeff = 16;
            }
            grid.x = m;
            grid.y = fold_coeff;
            block.x = k / (2 * fold_coeff);
            add_QKV_bias_opt<<<grid, block, 0, stream>>>((half*)Q, (half*)bias_Q, (half*)K, (half*)bias_K, (half*)V, (half*)bias_V, (half*)q_buf_, (half*)k_buf_, 
                (half*)v_buf_, batch_size, seq_len, head_num, size_per_head / 2);
    }
}


template void add_QKV_bias_opt_kernel<float>( void* Q, const void* bias_Q, void* K, const void* bias_K, void* V, const void* bias_V, void* q_buf_, void* k_buf_, void* v_buf_,
    const int& batch_size, const int& seq_len, const int& head_num, const int& size_per_head, const cudaStream_t stream);
template void add_QKV_bias_opt_kernel<half>(void* Q, const void* bias_Q,  void* K, const void* bias_K, void* V, const void* bias_V, void* q_buf_, void* k_buf_, void* v_buf_,
    const int& batch_size, const int& seq_len, const int& head_num, const int& size_per_head, const cudaStream_t stream);


template<typename T>
__global__
void fused_add_QKV_bias(T* QKV, const T* bias_Q, const T* bias_K, const T* bias_V, T* q_buf_, T* k_buf_, T* v_buf_,
                  const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    T* data_ptr;
    T* buf_ptr;
    const T* bias_ptr;

    int m = batch_size * seq_len;
    int n = head_num * size_per_head;
    int batch_id = (blockIdx.x  % m) / seq_len;
    int word_start_id = (blockIdx.x ) % seq_len;

    int qkv_id = blockIdx.x  /  m ;
    int row_offset = (blockIdx.x % m)  * n;
    int bid_offset = 2 * n * seq_len;
    if(qkv_id == 0)
    {
        data_ptr = QKV + row_offset + batch_id * bid_offset + 2 * word_start_id * n;
        buf_ptr = q_buf_;
        bias_ptr = bias_Q;
    }
    else if(qkv_id == 1)
    {
        data_ptr = QKV + row_offset + batch_id * bid_offset + 2 * word_start_id * n + n;
        buf_ptr = k_buf_;
        bias_ptr = bias_K;
    }
    else
    {
        data_ptr = QKV + row_offset + batch_id * bid_offset + 2 * word_start_id * n + 2 * n;
        buf_ptr = v_buf_;
        bias_ptr = bias_V;
    }

    int head_id = (threadIdx.x + blockIdx.y * blockDim.x) / size_per_head;
    int id_in_head = threadIdx.x % size_per_head;

    T bias = __ldg(&bias_ptr[threadIdx.x + blockDim.x * blockIdx.y]);

    for(int i = word_start_id; i < word_start_id + 1; ++i)
    {
        T tmp = data_ptr[threadIdx.x + blockDim.x * blockIdx.y] + bias;
        int target_id = batch_id * (seq_len * head_num * size_per_head) + head_id * seq_len * size_per_head +
                        i * size_per_head + id_in_head;

        buf_ptr[target_id] = tmp;
        data_ptr += n;
    }
}

__global__
void fused_add_QKV_bias( half* QKV, const half* bias_Q, const half* bias_K, const half* bias_V,
                  half* q_buf_, half* k_buf_, half* v_buf_,
                  const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int tid = blockIdx.x * (size_per_head * head_num) + threadIdx.x + blockDim.x * blockIdx.y;
    int batch_id = tid / (head_num * seq_len * size_per_head);
    int seq_id = (tid % (head_num * seq_len * size_per_head)) / (head_num * size_per_head);
    int head_id = (tid % (head_num * size_per_head)) / size_per_head;
    int id = tid % size_per_head;
    int target_id = target_index(batch_id, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);
    
    int bias_id = threadIdx.x + blockDim.x * blockIdx.y;

    int q_tid = tid + 2 * batch_id * size_per_head * head_num * seq_len + 2 * seq_id * size_per_head * head_num;
    int k_tid = tid + 2 * batch_id * size_per_head * head_num * seq_len + size_per_head * head_num + 2 * seq_id * size_per_head * head_num;
    int v_tid = tid + 2 * batch_id * size_per_head * head_num * seq_len + 2 * size_per_head * head_num + 2 * seq_id * size_per_head * head_num;


    half2* src_ptr = (half2*)QKV;
    half2* dst_ptr = (half2*)q_buf_;
    const half2* bias_ptr = (const half2*)bias_Q;
    dst_ptr[target_id] = __hadd2(src_ptr[q_tid],  __ldg(&bias_ptr[bias_id]));

    dst_ptr = (half2*)k_buf_;
    bias_ptr = (const half2*)bias_K;
    dst_ptr[target_id] = __hadd2(src_ptr[k_tid],  __ldg(&bias_ptr[bias_id]));

    dst_ptr = (half2*)v_buf_;
    bias_ptr = (const half2*)bias_V;
    dst_ptr[target_id] = __hadd2(src_ptr[v_tid],  __ldg(&bias_ptr[bias_id]));
}


template<typename T>
void fused_add_QKV_bias_kernel( void* QKV, const void* bias_Q,  const void* bias_K,  const void* bias_V, void* q_buf_, void* k_buf_, void* v_buf_,
                  const int& batch_size, const int& seq_len, const int& head_num, const int& size_per_head, const cudaStream_t stream){
        int m = batch_size * seq_len;
        int k = head_num * size_per_head;
        int qkv_types = 3;

        //assert(m * qkv_types <= 65536 && "batch_size * seq_len must <= 65536");
        int fold_coeff = 1;
        dim3 grid;
        dim3 block;

        if (sizeof(T) == sizeof(float)){
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
            grid.x = m * qkv_types;
            grid.y = fold_coeff;
            block.x = k / fold_coeff;
            fused_add_QKV_bias<<<grid, block, 0, stream>>>((float*)QKV, (float*)bias_Q, (float*)bias_K, (float*)bias_V, (float*)q_buf_, (float*)k_buf_, 
                (float*)v_buf_, batch_size, seq_len, head_num, size_per_head);
        }else{
            if (k <= 2048){
                fold_coeff = 2;
            }else if( k <= 4096){
                fold_coeff = 2;
            }else if(k <= 8192){
                fold_coeff = 4;
            }else if(k <= 16384){
                fold_coeff = 8;
            }else if(k <= 16384 * 2){
                fold_coeff = 16;
            }
            grid.x = m;
            grid.y = fold_coeff;
            block.x = k / (2 * fold_coeff);
            fused_add_QKV_bias<<<grid, block, 0, stream>>>((half*)QKV, (half*)bias_Q, (half*)bias_K, (half*)bias_V, (half*)q_buf_, (half*)k_buf_, 
                (half*)v_buf_, batch_size, seq_len, head_num, size_per_head / 2);
    }

}

template void fused_add_QKV_bias_kernel<float>( void* QKV, const void* bias_Q,  const void* bias_K,  const void* bias_V, void* q_buf_, void* k_buf_, void* v_buf_,
                  const int& batch_size, const int& seq_len, const int& head_num, const int& size_per_head, const cudaStream_t stream);

template void fused_add_QKV_bias_kernel<half>( void* QKV, const void* bias_Q,  const void* bias_K,  const void* bias_V, void* q_buf_, void* k_buf_, void* v_buf_,
                  const int& batch_size, const int& seq_len, const int& head_num, const int& size_per_head, const cudaStream_t stream);

