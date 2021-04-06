void gen_mask_offset_kernel(const int64_t *every_seq_len_inbatch,
                            const int current_batch_num, int64_t *dest, const cudaStream_t stream);

void fill_kernel(int64_t *data_ptr, int64_t size, int64_t val);

int reduce_kernel(int64_t *data_ptr, int64_t size);

void compute_len_inbatch_kernel(int64_t *data_ptr, int batch_size, int seq_len, int64_t *dest, const cudaStream_t stream);
