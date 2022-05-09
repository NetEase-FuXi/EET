template <typename T>
void add_bias_input_layernorm_kernel(void *output, const void *input,
                                     const void *bias, const void *gamma,
                                     const void *beta, const int &m, int &n,
                                     const cudaStream_t stream);
template <typename T>
void layernorm(const void *input, const void *gamma,
               const void *beta, void *output,
               const int &m, const int &n,
               const cudaStream_t stream);

template <typename T>
void T5layernorm(const void *input, const void *gamma,
                 void *output, const int &m, const int &n,
                 const cudaStream_t stream);