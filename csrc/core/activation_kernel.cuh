template <typename T>
void add_bias_act_kernel(void* ffn_inner, const void* bias, int m, int n ,const int act_type ,const cudaStream_t stream);