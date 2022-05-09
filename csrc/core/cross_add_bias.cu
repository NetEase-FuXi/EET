#include "core/cross_add_bias.cuh"
#include <assert.h>

template<typename T>
__global__
void add_Q_bias_opt(T* Q, const T* bias_Q, T* q_buf_,
                  const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{

    T* data_ptr;
    T* buf_ptr;
    const T* bias_ptr;

    int m = batch_size * seq_len;
    int n = head_num * size_per_head;

    int row_offset = (blockIdx.x  % m)  * n;

    data_ptr = Q + row_offset;
    buf_ptr = q_buf_;
    bias_ptr = bias_Q;


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

template <typename T>
__global__ 
void Q_transpose_opt(T *Q, T *q_buf_, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{

    T* data_ptr;
    T* buf_ptr;

    int m = batch_size * seq_len;
    int n = head_num * size_per_head;

    int row_offset = (blockIdx.x  % m)  * n;

    data_ptr = Q + row_offset;
    buf_ptr = q_buf_;

    int batch_id = (blockIdx.x  % m) / seq_len;
    int head_id = (threadIdx.x + blockIdx.y * blockDim.x) / size_per_head;
    int id_in_head = threadIdx.x % size_per_head;
    int word_start_id = (blockIdx.x ) % seq_len;

    for(int i = word_start_id; i < word_start_id + 1; ++i)
    {
        T tmp = data_ptr[threadIdx.x + blockDim.x * blockIdx.y];

        int target_id = batch_id * (seq_len * head_num * size_per_head) + head_id * seq_len * size_per_head +
                        i * size_per_head + id_in_head;

        buf_ptr[target_id] = tmp;
        data_ptr += n;
    }
}


__inline__ __device__
int target_index(int id1, int id2, int id3, int id4, int dim_1, int dim_2, int dim_3, int dim_4)
{
  return id1 * (dim_2 * dim_3 * dim_4) + id3 * (dim_2 * dim_4) + id2 * dim_4 + id4;
}

template<>
__global__
void add_Q_bias_opt<half>(half* Q, const half* bias_Q, half* q_buf_, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
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
}

template<>
__global__
void Q_transpose_opt<half>(half* Q, half* q_buf_, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int tid = blockIdx.x * (size_per_head * head_num) + threadIdx.x + blockDim.x * blockIdx.y;
    int batch_id = tid / (head_num * seq_len * size_per_head);
    int seq_id = (tid % (head_num * seq_len * size_per_head)) / (head_num * size_per_head);
    int head_id = (tid % (head_num * size_per_head)) / size_per_head;
    int id = tid % size_per_head;
    int target_id = target_index(batch_id, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);

    half2* src_ptr = (half2*)Q;
    half2* dst_ptr = (half2*)q_buf_;
    dst_ptr[target_id] = src_ptr[tid];
}


template<typename T>
__global__
void add_KV_bias_opt( T* K, const T* bias_K, T* V, const T* bias_V,T* k_buf_, T* v_buf_,
                  const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{

    T* data_ptr;
    T* buf_ptr;
    const T* bias_ptr;

    int m = batch_size * seq_len;
    int n = head_num * size_per_head;

    int qkv_id = blockIdx.x  /  m ;
    int row_offset = (blockIdx.x  % m)  * n;

    if(qkv_id == 0)
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

template <typename T>
__global__ 
void KV_transpose_opt(T *K, T *V, T *k_buf_, T *v_buf_, const int batch_size, const int seq_len, 
                      const int head_num, const int size_per_head)
{
    T* data_ptr;
    T* buf_ptr;

    int m = batch_size * seq_len;
    int n = head_num * size_per_head;

    int qkv_id = blockIdx.x  /  m ;
    int row_offset = (blockIdx.x  % m)  * n;

    if(qkv_id == 0)
    {
        data_ptr = K + row_offset;
        buf_ptr = k_buf_;
    }
    else
    {
        data_ptr = V + row_offset;
        buf_ptr = v_buf_;
    }


    int batch_id = (blockIdx.x  % m) / seq_len;
    int head_id = (threadIdx.x + blockIdx.y * blockDim.x) / size_per_head;
    int id_in_head = threadIdx.x % size_per_head;
    int word_start_id = (blockIdx.x ) % seq_len;

    for(int i = word_start_id; i < word_start_id + 1; ++i)
    {
        T tmp = data_ptr[threadIdx.x + blockDim.x * blockIdx.y];

        int target_id = batch_id * (seq_len * head_num * size_per_head) + head_id * seq_len * size_per_head +
                        i * size_per_head + id_in_head;

        buf_ptr[target_id] = tmp;
        data_ptr += n;
    }
}

template<>
__global__
void add_KV_bias_opt<half>(half* K, const half* bias_K, half* V, const half* bias_V, half* k_buf_, half* v_buf_, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int tid = blockIdx.x * (size_per_head * head_num) + threadIdx.x + blockDim.x * blockIdx.y;
    int batch_id = tid / (head_num * seq_len * size_per_head);
    int seq_id = (tid % (head_num * seq_len * size_per_head)) / (head_num * size_per_head);
    int head_id = (tid % (head_num * size_per_head)) / size_per_head;
    int id = tid % size_per_head;
    int target_id = target_index(batch_id, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);

    int bias_id = threadIdx.x + blockDim.x * blockIdx.y;

    half2* src_ptr = (half2*)K;
    half2* dst_ptr = (half2*)k_buf_;
    const half2* bias_ptr = (const half2*)bias_K;
    dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));

    src_ptr = (half2*)V;
    dst_ptr = (half2*)v_buf_;
    bias_ptr = (const half2*)bias_V;
    dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));
}

template<>
__global__
void KV_transpose_opt<half>(half* K, half* V, half* k_buf_, half* v_buf_, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int tid = blockIdx.x * (size_per_head * head_num) + threadIdx.x + blockDim.x * blockIdx.y;
    int batch_id = tid / (head_num * seq_len * size_per_head);
    int seq_id = (tid % (head_num * seq_len * size_per_head)) / (head_num * size_per_head);
    int head_id = (tid % (head_num * size_per_head)) / size_per_head;
    int id = tid % size_per_head;
    int target_id = target_index(batch_id, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);

    half2* src_ptr = (half2*)K;
    half2* dst_ptr = (half2*)k_buf_;
    dst_ptr[target_id] = src_ptr[tid];

    src_ptr = (half2*)V;
    dst_ptr = (half2*)v_buf_;
    dst_ptr[target_id] = src_ptr[tid];
}

template<typename T>
void add_QKV_bias_cross_opt_kernel(void* Q, const void* bias_Q, void* K, const void* bias_K, void* V, const void* bias_V, void* q_buf_, void* k_buf_, void* v_buf_,
                  const int& batch_size, const int& seq_len, const int& mem_seq_len,const int& head_num, const int& size_per_head, const cudaStream_t stream){
        int qkv_types = 1;
        int m = batch_size * seq_len;
        int k = head_num * size_per_head;
        assert(m * qkv_types * 3 <= 65536 && "batch_size * seq_len must <= 65536"); 
        int fold_coeff = 1;
        bool is_add_bias = bias_Q != nullptr;

        dim3 grid;
        dim3 block;
        //TODO - int8
        if (sizeof(T) == sizeof(float)) {
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
            if (is_add_bias) {
                add_Q_bias_opt<<<grid, block, 0, stream>>>((T*)Q, (T*)bias_Q, (T*)q_buf_, batch_size, seq_len, head_num, size_per_head);
            } else {
                Q_transpose_opt<<<grid, block, 0, stream>>>((T*)Q, (T*)q_buf_, batch_size, seq_len, head_num, size_per_head);
            }

            // mem_len != seq_len 
            qkv_types = 2;
            m = batch_size * mem_seq_len;
            grid.x = m * qkv_types;

            if (is_add_bias) {
                add_KV_bias_opt<<<grid, block, 0, stream>>>((T*)K,(T*)bias_K, (T*)V, (T*)bias_V,(T*)k_buf_, (T*)v_buf_, batch_size, mem_seq_len, head_num, size_per_head);
            } else {
                KV_transpose_opt<<<grid, block, 0, stream>>>((T*)K, (T*)V, (T*)k_buf_, (T*)v_buf_, batch_size, mem_seq_len, head_num, size_per_head);                 
            }
        } else {
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
            grid.x = m * qkv_types;
            grid.y = fold_coeff;
            block.x = k / (2 * fold_coeff);
            if (is_add_bias) {
                add_Q_bias_opt<<<grid, block, 0, stream>>>((T*)Q, (T*)bias_Q, (T*)q_buf_, batch_size, seq_len, head_num, size_per_head / 2);
            } else {
                Q_transpose_opt<<<grid, block, 0, stream>>>((T*)Q, (T*)q_buf_, batch_size, seq_len, head_num, size_per_head / 2);
            }
            
            // memory
            qkv_types = 2;
            m = batch_size * mem_seq_len;
            grid.x = m * qkv_types;

            if (is_add_bias) {
                add_KV_bias_opt<<<grid, block, 0, stream>>>((T*)K, (T*)bias_K, (T*)V, (T*)bias_V, (T*)k_buf_, (T*)v_buf_, batch_size, mem_seq_len, head_num, size_per_head / 2);
            } else {
                KV_transpose_opt<<<grid, block, 0, stream>>>((T*)K, (T*)V, (T*)k_buf_, (T*)v_buf_, batch_size, mem_seq_len, head_num, size_per_head / 2);
            }
    }
}


template void add_QKV_bias_cross_opt_kernel<float>(void* Q, const void* bias_Q, void* K, const void* bias_K, void* V, const void* bias_V, void* q_buf_, void* k_buf_, void* v_buf_,
                  const int& batch_size, const int& seq_len, const int& mem_seq_len,const int& head_num, const int& size_per_head, const cudaStream_t stream);
template void add_QKV_bias_cross_opt_kernel<half>(void* Q, const void* bias_Q, void* K, const void* bias_K, void* V, const void* bias_V, void* q_buf_, void* k_buf_, void* v_buf_,
                  const int& batch_size, const int& seq_len, const int& mem_seq_len,const int& head_num, const int& size_per_head, const cudaStream_t stream);
    
    