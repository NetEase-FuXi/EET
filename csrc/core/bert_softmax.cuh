template <class T>
void bert_softmax_kernel(void *qk_buf, void* attr_mask, const int& batch_size,
                    const int& head_num, const int& seq_len, const float& scalar, const cudaStream_t stream);