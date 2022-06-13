template <typename T>
void masked_attention_dispatch(void *key_buf, void *value_buf,
                               void *query_buf, const void *self_Q_bias,
                               void *key_cache, const void *self_K_bias, void *value_cache, const void *self_V_bias,
                               void *context_buf, int &batch_size, int &first_batch_size, int &head_num, int &size_per_head, const int &step, cudaStream_t stream, const int64_t *pre_padding_len, const int64_t *reorder_index);

template <typename T>
void cross_attention_dispatch(void *query_buf, const void *Q_bias,
                              void *key_cache, const void *K_bias, void *value_cache, const void *V_bias, const int *length,
                              void *context_buf, int &batch_size, int &head_num, int &size_per_head, int &step, int &seq_len, cudaStream_t stream);

template <typename T>
void fused_masked_attention_dispatch(void *qkv_buf, const void *self_Q_bias,
                                     void *key_cache, const void *self_K_bias, void *value_cache, const void *self_V_bias,
                                     void *context_buf, int &batch_size, int &first_batch_size, int &head_num, int &size_per_head, const int &step, cudaStream_t stream, const int64_t *pre_padding_len, const int64_t *reorder_index, const void *position_bias);
