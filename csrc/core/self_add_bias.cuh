#include <cuda_runtime.h>
#include <cuda_fp16.h>
//#include "op/common.hpp"
template<typename T>
void add_QKV_bias_opt_kernel( void* Q, const void* bias_Q, void* K, const void* bias_K, void* V, const void* bias_V, void* q_buf_, void* k_buf_, void* v_buf_,
                  const int& batch_size, const int& seq_len, const int& head_num, const int& size_per_head, const cudaStream_t stream);

template<typename T>
void add_QKV_bias_rebuild_padding_kernel( void* Q, const void* bias_Q, void* K, const void* bias_K, void* V, const void* bias_V, void* q_buf_, void* k_buf_, void* v_buf_,
                  int valid_word_num, int64_t *sequence_id_offset ,const int& batch_size, const int& seq_len, const int& head_num, const int& size_per_head, const cudaStream_t stream);


template<typename T>
void fused_add_QKV_bias_kernel( void* QKV, const void* bias_Q,  const void* bias_K,  const void* bias_V, void* q_buf_, void* k_buf_, void* v_buf_,
    const int& batch_size, const int& seq_len, const int& head_num, const int& size_per_head, const cudaStream_t stream);
                      