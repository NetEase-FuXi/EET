#include "op/layer_norm.hpp"
#include "core/layer_norm.cuh"

namespace eet
{
    namespace op
    {
        LayerNorm::LayerNorm(MetaDesc desc, const torch::Tensor& gamma, const torch::Tensor& beta):
            desc_(desc),
            layernorm_weights_(gamma.data_ptr()),
            layernorm_bias_(beta.data_ptr()){
            output_ = torch::zeros({desc_.batch_size_ , desc_.max_seq_len_ ,desc_.hidden_units_}, desc_.options_);
        }

        // layerNorm
        torch::Tensor LayerNorm::layer_norm(const torch::Tensor& input_tensor)
        {
            int cur_batch_size = input_tensor.sizes()[0];
            int cur_seq_len = input_tensor.sizes()[1];
            const int m = cur_batch_size * cur_seq_len;
            int n = desc_.hidden_units_;
            if (layernorm_bias_ != nullptr)
            {
                RUN_KERNEL(layernorm, desc_.dtype_, input_tensor.data_ptr(), layernorm_weights_, layernorm_bias_, output_.data_ptr(), m, n, desc_.layernorm_eps_, desc_.stream);
            }
            else
            {
                RUN_KERNEL(RMSnorm, desc_.dtype_, input_tensor.data_ptr(), layernorm_weights_, output_.data_ptr(), m, n, desc_.layernorm_eps_, desc_.stream);
            }
            auto res = torch::from_blob(output_.data_ptr(), input_tensor.sizes(), input_tensor.strides(), desc_.options_);
            return std::move(res);
            // return output_;
        }

    } // namespace op
} // namespace eet