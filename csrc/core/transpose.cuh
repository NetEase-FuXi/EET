template <typename T>
void transpose_kernel(void* transpose_dst, void* dst, const int& batch_size, const int& seq_len,
                const int& head_num, const int& size_per_head, const cudaStream_t stream);

template <typename T>
void copyKV_transpose_kernel(void* d_K_buf, void* d_V_buf,void* K_buf, void* V_buf,const int& batch_size, const int& seq_len,
                    const int& head_num, const int& size_per_head);

template <typename T>
void copyKV_transpose_cross_kernel(void* d_K_buf, void* d_V_buf,void* K_buf, void* V_buf,const int& batch_size, const int& seq_len,
                                      const int& head_num, const int& size_per_head);