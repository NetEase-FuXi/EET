template <typename T>
void add_bias_kernel(void* out, const void* bias, int m, int n, const cudaStream_t stream);

template <typename T>
void add_bias_input_kernel(void* output, const void* input, const void* bias,const int m, const int n, const cudaStream_t stream);
