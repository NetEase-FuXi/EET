template <class T>
void softmax_kernel(void *qk_buf, const int64_t* padding_index, const int& batch_size,
                    const int& head_num, const int& seq_len, const cudaStream_t stream);

template <class T>
void launch_masked_softmax_kernel(void *qk_buf, void *position_bias, const int64_t *padding_index, const int batch_size,
                                  const int head_num, const int seq_len, const cudaStream_t stream);