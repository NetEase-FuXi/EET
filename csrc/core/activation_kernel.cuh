template <typename T>
void add_bias_act_kernel(void* ffn_inner, const void* bias, int m, int n ,const int act_type ,const cudaStream_t stream);

template <typename T>
void gated_act_kernel(void* inner_act, void* inner_linear, int m, int n , const int act_type ,const cudaStream_t stream);