template <typename T>
void add_bias_kernel(void* out, const void* bias, int m, int n, const cudaStream_t stream);

template <typename T>
void add_bias_input_kernel(void* output, const void* input, const void* bias,const int m, const int n, const cudaStream_t stream);

template <class T>
void add_relative_attn_bias_kernel(void *qk_buf, const void *relative_attention_bias, const int &batch_size, const int &head_num, 
                                   const int &seq_len, const cudaStream_t stream);