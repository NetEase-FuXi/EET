#ifndef _OP_GATEDFFN_INT8_HPP_
#define _OP_GATEDFFN_INT8_HPP_

#include "op/common.hpp"
#include "op/meta_desc.hpp"
#include "op/mmanager.hpp"

namespace eet{
    namespace op{

        class GatedFeedForwardNetworkInt8 : public OpBase{
        public:
            GatedFeedForwardNetworkInt8(MetaDesc desc,
                            const torch::Tensor& Intermediate_0_weights,
                            const torch::Tensor& Intermediate_0_scale,
                            const torch::Tensor& Intermediate_0_bias,
                            const torch::Tensor& Intermediate_1_weights,
                            const torch::Tensor& Intermediate_1_scale,
                            const torch::Tensor& Intermediate_1_bias,
                            const torch::Tensor& Output_weights,
                            const torch::Tensor& Output_scale,
                            const torch::Tensor& Output_bias,
                            const torch::Tensor& layernorm_weights,
                            const torch::Tensor& layernorm_bias,
                            const std::string &ffn_cache_name = "out_cache");

            torch::Tensor forward(torch::Tensor& input, 
                                    bool pre_layernorm,
                                    bool add_residual);

            ~GatedFeedForwardNetworkInt8(){
            };
            
        private:
            void layer_norm(const torch::Tensor& input_tensor, Buffer& layernorm_tensor);

            void dequant1(Buffer& weight_buffer);

            void dequant2(Buffer& weight_buffer);
            
            void dequant3(Buffer& weight_buffer);

            void fc1_mul(void* input,  Buffer& ffn_inner, Buffer& weight_buf);

            void fc2_mul(void* input,  Buffer& ffn_inner, Buffer& weight_buf);

            void gated_act(Buffer& inner_act, Buffer& inner_linear);

            void add_bias_act(Buffer& ffn_inner);

            void fc3_mul(const Buffer& ffn_inner, Buffer& output, Buffer& weight_buf);

            void add_input_bias_layernorm(Buffer& output,torch::Tensor& input_tensor,bool pre_layernorm, bool add_residual);

            void fc1_mul_opt(void* input,  Buffer& ffn_inner);
            void fc2_mul_opt(void* input,  Buffer& ffn_inner);
            void fc3_mul_opt(const Buffer& ffn_inner, Buffer& output);
            MetaDesc desc_;
            // torch::Tensor output_;

            cublasGemmAlgo_t fc1_algo_, fc2_algo_, fc3_algo_;
            int act_type_;
            int cur_batch_size_;
            int cur_seq_len_;
            int size_per_head_;
            int d_ff_;

            void* alpha_;
            void* beta_;
            void* intermediate_0_scale_;
            void* intermediate_1_scale_;
            void* output_scale_;

            void* intermediate_0_weights_;
            void* intermediate_0_bias_;
            void* intermediate_1_weights_;
            void* intermediate_1_bias_;
            void* output_weights_;
            void* output_bias_;
            void* layernorm_weights_;
            void* layernorm_bias_;
            std::string ffn_cache_name_;
        };
    }
}
#endif