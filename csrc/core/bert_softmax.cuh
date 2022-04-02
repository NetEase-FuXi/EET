template <class T>
void bert_softmax_kernel(void *qk_buf, const int64_t *pre_padding_len, const int &batch_size, const int &head_num,
                         const int &seq_len, const float &scalar, bool need_sequence_mask, const cudaStream_t stream);