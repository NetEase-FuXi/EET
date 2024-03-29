#ifndef _OP_BAICHUAN_MMHA_HPP_
#define _OP_BAICHUAN_MMHA_HPP_

#include "op/common.hpp"
#include "op/mmanager.hpp"
#include "op/meta_desc.hpp"

namespace eet{
    namespace op{

        class BaichuanMmha : public OpBase{
        public:
            BaichuanMmha(MetaDesc desc,
                                    const torch::Tensor& QKV_weights,
                                    const torch::Tensor& QKV_scale,
                                    const torch::Tensor& Q_bias,
                                    const torch::Tensor& K_bias,
                                    const torch::Tensor& V_bias,
                                    const torch::Tensor& Output_weights,
                                    const torch::Tensor& Output_scale,
                                    const torch::Tensor& Output_bias,
                                    const torch::Tensor& layernorm_weights,
                                    const torch::Tensor& layernorm_bias);

            torch::Tensor forward(torch::Tensor& input, 
                                    const torch::Tensor& pre_padding_len,
                                    const torch::Tensor& reorder_state,
                                    bool pre_layernorm,
                                    bool add_residual,
                                    bool first_pass,
                                    const torch::Tensor &relative_attention_bias);

            // full decode
            torch::Tensor forward_full(torch::Tensor& input, 
                                    const torch::Tensor& pre_padding_length,
                                    bool pre_layernorm,
                                    bool add_residual,
                                    const torch::Tensor &relative_attention_bias = torch::empty(0));

            // incremental decode
            torch::Tensor forward_inc(torch::Tensor& input, 
                                    const torch::Tensor& pre_padding_len,
                                    const torch::Tensor& reorder_state,
                                    bool pre_layernorm,
                                    bool add_residual,
                                    const torch::Tensor &relative_attention_bias = torch::empty(0));

            ~BaichuanMmha(){
                // check_cuda_error(cudaFree(&fused_qkv_ptr_));
            };
            
        private:
            // TODO layernorm
            void layer_norm(const torch::Tensor& input_tensor,
                                    Buffer& layernorm_query);

            void qkv_weights_mul(void* input, 
                                Buffer& qkv_buffer);

            void qkv_weights_mul_weight_only_int8(void* input, 
                                Buffer& qkv_buffer);

            void qkv_add_bias(const Buffer &qkv_buffer,
                              Buffer &q_buf,
                              Buffer &k_buf,
                              Buffer &v_buf);

            void q_k_mul(const Buffer& q_buf, const Buffer& k_buf, 
                            Buffer& qk_buf);

            void qk_softmax(Buffer& qk_buf, void* relative_attention_bias, const int64_t *padding_len);

            void attn_v_mul(const Buffer& qk_buf, const Buffer& v_buf,
                            Buffer& transpose_dst);

            void transpose(const Buffer& transpose_dst, Buffer&  dst);

            void project(const Buffer &dst,
                         Buffer &res,
                         torch::Tensor &input,
                         bool pre_layernorm,
                         bool add_residual);

            void fused_masked_attention(const Buffer &qkv_buffer,
                                        Buffer &context_buf,
                                        const int64_t *padding_len,
                                        const int64_t *reorder_index,
                                        const void *relative_attention_bias_);

            void masked_attention(const Buffer &q_buf,
                                  const Buffer &k_buf,
                                  const Buffer &v_buf,
                                  Buffer &context_buf,
                                  const int64_t *padding_len,
                                  const int64_t *reorder_index);

            void kv_transpose(torch::Tensor& d_K_buf, torch::Tensor& d_V_buf,Buffer& K_buf,Buffer& V_buf);

            void reorder_cache(torch::Tensor &K_cache, torch::Tensor &V_cache, Buffer &K_buf, Buffer &V_buf, const int64_t *reorder_index);
            void reorder_cache(torch::Tensor &K_cache, torch::Tensor &V_cache, torch::Tensor &K_buf, torch::Tensor &V_buf, const int64_t *reorder_index);
            
            MetaDesc desc_;
            // torch::Tensor output_;
            torch::Tensor k_cache_, v_cache_;
            cublasGemmAlgo_t qkv_weights_algo_, q_k_algo_, attn_v_algo_;

            bool with_bias_;
            int cur_batch_size_;
            int first_batch_size_;
            int cur_seq_len_;
            int size_per_head_;
            int inner_dim_;
            int step_;
            void* alpha_;
            void* beta_;
            void* atten_scaler_;
            void** qkv_input_;
            void** qkv_kernel_;
            void** qkv_buf_;

        private:
            void* qkv_weights_;
            void* qkv_scale_;
            void* q_bias_;
            void* k_bias_;
            void* v_bias_;
            void* output_weights_;
            void* output_scale_;
            void* output_bias_;
            void* layernorm_weights_;
            void* layernorm_bias_;

        };
    }
}
#endif
