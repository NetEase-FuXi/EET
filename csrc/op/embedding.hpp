#ifndef _OP_EMBEDDING_HPP_
#define _OP_EMBEDDING_HPP_

#include "op/common.hpp"
#include "op/meta_desc.hpp"
#include "op/mmanager.hpp"

namespace eet
{
    namespace op
    {
        class Embedding : public OpBase
        {
        public:
            Embedding(MetaDesc desc, 
                    const torch::Tensor& embedding_weights, 
                    const torch::Tensor& position_weights, 
                    const torch::Tensor& token_type_weights,                    
                    const torch::Tensor& layernorm_weights, 
                    const torch::Tensor& layernorm_bias);

            torch::Tensor forward_transformers(const torch::Tensor& input_tensor,const torch::Tensor &position_ids,const torch::Tensor &token_type_ids, bool if_layernorm);
            torch::Tensor forward_fairseq(const torch::Tensor& input_tensor,const torch::Tensor &positions_mask,bool no_scale_embedding, int padding_idx,bool first_pass);

        private:
            void fused_embedding(const torch::Tensor& input_tensor,const torch::Tensor &position_ids,const torch::Tensor &token_type_ids, void* embedding_out); 

            void layer_norm(Buffer& input,Buffer& out); 

            int step_;
            MetaDesc desc_;
            // torch::Tensor output_;
            void *embedding_weights_;
            void *position_weights_;
            void *token_type_weights_;
            void *layernorm_weights_;
            void *layernorm_bias_;
            int cur_batch_size_;
            int cur_seq_len_;
        };
    } // namespace op
} // namespace eet
#endif