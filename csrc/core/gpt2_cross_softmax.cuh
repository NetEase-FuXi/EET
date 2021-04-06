template <class T>
void cross_softmax_kernel(void *qk_buf, const int& batch_size,
                    const int& head_num, const int& seq_len,const int& mem_seq_len, const float& scalar, const cudaStream_t stream);
