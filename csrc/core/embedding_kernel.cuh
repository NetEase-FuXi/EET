template <typename T>
void embedding_lookup_kernel(const void* embedding_table, const int64_t *input_ids, void* embedding_lookup_tensor,
  const int num_id, const int hidden_units, cudaStream_t stream,bool ifacc,const int ratio,bool no_scale_embedding);

template <typename T>
void position_encoding_kernel(void* output,const int64_t* positions, int seq_len,  int step,int padding_idx,int m, int n, cudaStream_t stream);
