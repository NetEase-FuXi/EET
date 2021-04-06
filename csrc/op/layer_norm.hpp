#ifndef _OP_LAYER_NORM_HPP_
#define _OP_LAYER_NORM_HPP_

#include "op/common.hpp"
#include "op/meta_desc.hpp"
#include "op/mmanager.hpp"

namespace eet
{
    namespace op
    {
        class LayerNorm
        {
        public:
            LayerNorm(MetaDesc desc, const torch::Tensor& gamma, const torch::Tensor& beta);
            torch::Tensor layer_norm(const torch::Tensor& input_tensor);
        private:
            MetaDesc desc_;
            torch::Tensor output_;
            void *layernorm_weights_;
            void *layernorm_bias_;
        };
    } // namespace op
} // namespace eet
#endif