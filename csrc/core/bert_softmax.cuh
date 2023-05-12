template <class T>
void launch_softmax_kernel(void *qk_buf, void *position_bias, const int64_t *padding_len, const int batch_size,
                           const int head_num, const int seq_len, bool need_sequence_mask, const cudaStream_t stream);