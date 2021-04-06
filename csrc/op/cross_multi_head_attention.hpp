#ifndef _OP_CROSS_MULTI_HEAD_ATTENTION_HPP_
#define _OP_CROSS_MULTI_HEAD_ATTENTION_HPP_

#include "op/common.hpp"
#include "op/meta_desc.hpp"
#include "op/mmanager.hpp"

namespace eet{
    namespace op{

        class CrossMultiHeadAttention{
        public:
            CrossMultiHeadAttention(MetaDesc desc,
                                    const torch::Tensor& Q_weights,
                                    const torch::Tensor& Q_bias,
                                    const torch::Tensor& K_weights,
                                    const torch::Tensor& K_bias,
                                    const torch::Tensor& V_weights,
                                    const torch::Tensor& V_bias,
                                    const torch::Tensor& Output_weights,
                                    const torch::Tensor& Output_bias,
                                    const torch::Tensor& layernorm_weights,
                                    const torch::Tensor& layernorm_bias);

            torch::Tensor forward(torch::Tensor& input,
                                    torch::Tensor& memory,
                                    const torch::Tensor& padding_index,
                                    bool pre_layernorm,
                                    bool add_redusial,
                                    const torch::Tensor& length_per_sample,
                                    bool first_pass);
        

            // full decode
            torch::Tensor forward_full(torch::Tensor& input, 
                                    torch::Tensor& memory,
                                    const torch::Tensor& padding_index,
                                    bool pre_layernorm,
                                    bool add_redusial);
            
            // incremental decode
            torch::Tensor forward_inc(torch::Tensor& input,
                                    torch::Tensor& memory,
                                    bool pre_layernorm,
                                    bool add_redusial,
                                    const torch::Tensor& length_per_sample);

            ~CrossMultiHeadAttention(){
            };
            
        private:
            void layer_norm(const torch::Tensor& input_tensor,
                        Buffer& layernorm_query);

            void qkv_weights_mul(void* input, 
                                    torch::Tensor& memory,
                                    Buffer& q_buffer,
                                    Buffer& k_buffer,
                                    Buffer& v_buffer);

            void qkv_add_bias(const Buffer& q_buffer,
                                    const Buffer& k_buffer,
                                    const Buffer& v_buffer,
                                    Buffer& q_buf,
                                    Buffer& k_buf,
                                    Buffer& v_buf);

            void q_k_mul(const Buffer& q_buf, const Buffer& k_buf, 
                            Buffer& qk_buf);

            void qk_softmax(Buffer& qk_buf, const torch::Tensor& padding_index);

            void attn_v_mul(const Buffer& qk_buf, const Buffer& v_buf,
                            Buffer& transpose_dst);

            void transpose(const Buffer& transpose_dst, Buffer&  dst);

            void project(const Buffer& dst, 
                    torch::Tensor& res,
                    torch::Tensor& input, 
                    bool pre_layernorm,
                    bool add_redusial);

            void attention_dispatch(const Buffer& q_buffer,
                                const torch::Tensor& length_per_sample,
                                Buffer& context_buf);

            void kv_transpose(torch::Tensor& d_K_buf, torch::Tensor& d_V_buf,Buffer& K_buf,Buffer& V_buf);
            MetaDesc desc_;
            torch::Tensor output_;
            torch::Tensor key_mem_cache_, value_mem_cache_;

            cublasGemmAlgo_t qkv_weights_algo_, q_k_algo_, attn_v_algo_;

            int cur_batch_size_;
            int cur_seq_len_;
            int mem_seq_len_;
            int size_per_head_;
            int step_;
            void* alpha_;
            void* beta_;

        private:
            void* q_weights_;
            void* k_weights_;
            void* v_weights_;
            void* q_bias_;
            void* k_bias_;
            void* v_bias_;
            void* output_weights_;
            void* output_bias_;
            void* layernorm_weights_;
            void* layernorm_bias_;
        };
    }
}
#endif