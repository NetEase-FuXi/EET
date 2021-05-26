#ifndef _OP_FFN_HPP_
#define _OP_FFN_HPP_

#include "op/common.hpp"
#include "op/meta_desc.hpp"
#include "op/mmanager.hpp"

namespace eet{
    namespace op{

        class FeedForwardNetwork : public OpBase{
        public:
            FeedForwardNetwork(MetaDesc desc,
                            const torch::Tensor& Intermediate_weights,
                            const torch::Tensor& Intermediate_bias,
                            const torch::Tensor& Output_weights,
                            const torch::Tensor& Output_bias,
                            const torch::Tensor& layernorm_weights,
                            const torch::Tensor& layernorm_bias);

            torch::Tensor forward(torch::Tensor& input, 
                                    bool pre_layernorm,
                                    bool add_redusial);

            ~FeedForwardNetwork(){
            };
            
        private:
            void layer_norm(const torch::Tensor& input_tensor, Buffer& layernorm_tensor);

            void fc1_mul(void* input,  Buffer& ffn_inner);

            void add_bias_act(Buffer& ffn_inner);

            void fc2_mul(const Buffer& ffn_inner, Buffer& output);

            void add_input_bias_layernorm(Buffer& output,torch::Tensor& input_tensor,bool pre_layernorm, bool add_redusial);

            MetaDesc desc_;
            // torch::Tensor output_;

            cublasGemmAlgo_t fc1_algo_, fc2_algo_;
            int act_type_;
            int cur_batch_size_;
            int cur_seq_len_;
            int size_per_head_;

            void* alpha_;
            void* beta_;

            void* intermediate_weights_;
            void* intermediate_bias_;
            void* output_weights_;
            void* output_bias_;
            void* layernorm_weights_;
            void* layernorm_bias_;
        };
    }
}
#endif