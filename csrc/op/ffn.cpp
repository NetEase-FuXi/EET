#include "op/ffn.hpp"
#include "core/add_bias.cuh"
#include "core/layer_norm.cuh"
#include "core/gpt2_self_softmax.cuh"
#include "core/activation_kernel.cuh"

namespace eet
{
    namespace op
    {
        FeedForwardNetwork::FeedForwardNetwork(MetaDesc desc,
                            const torch::Tensor& Intermediate_weights,
                            const torch::Tensor& Intermediate_bias,
                            const torch::Tensor& Output_weights,
                            const torch::Tensor& Output_bias,
                            const torch::Tensor& layernorm_weights,
                            const torch::Tensor& layernorm_bias) : 
        desc_(desc),
        intermediate_weights_(Intermediate_weights.data_ptr()),
        intermediate_bias_(Intermediate_bias.data_ptr()),
        output_weights_(Output_weights.data_ptr()),
        output_bias_(Output_bias.data_ptr()),
        layernorm_weights_(layernorm_weights.data_ptr()),
        layernorm_bias_(layernorm_bias.data_ptr())
        {   
            // Currently only supports gelu and relu
            if (desc_.activation_fn_ == "gelu" || desc_.activation_fn_ == "gelu_new" || desc_.activation_fn_ == "gelu_fast")
            {
                act_type_ = 1;
            }
            else if(desc_.activation_fn_ == "relu")
            {
                // relu
                act_type_ = 0;
            }
            else
            {
                std::cout << "unsupported activation " << std::endl;
                return;
            }
            size_per_head_ = desc_.hidden_units_ / desc_.head_num_;
            // output_ = torch::zeros({desc_.batch_size_, desc_.max_full_seq_len_, desc_.hidden_units_}, desc_.options_);

            switch (desc_.dtype_)
            {
            case torch::kFloat32:
                fc1_algo_ = CUBLAS_GEMM_DEFAULT;
                fc2_algo_ = CUBLAS_GEMM_DEFAULT;
                alpha_ = new float();
                beta_ = new float();
                *((float *)alpha_) = 1.0f;
                *((float *)beta_) = 0.0f;
                break;
            case torch::kFloat16:
                fc1_algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
                fc2_algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
                alpha_ = new half();
                beta_ = new half();
                *((half *)alpha_) = (half)1.0f;
                *((half *)beta_) = (half)0.0f;
                break;
            //TODO
            case torch::kInt8:
                break;
            }
        }

        torch::Tensor FeedForwardNetwork::forward(torch::Tensor &input,
                                                    bool pre_layernorm,
                                                    bool add_redusial)
        {
            assert((input.dtype() == desc_.dtype_) && "input's dtype is not the same as FeedForwardNetwork's dtype");
            cur_batch_size_ = input.sizes()[0];
            cur_seq_len_ = input.sizes()[1];

            //ffn_inner
            Buffer &ffn_inner = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                                                                        desc_.hidden_units_ * 4,
                                                                    desc_.dtype_, desc_.options_);

            if(pre_layernorm)
            {
                // pre_layerNorm
                Buffer& layernorme_tensor = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                        desc_.hidden_units_, desc_.dtype_, desc_.options_);
                layer_norm(input,layernorme_tensor);

                fc1_mul(layernorme_tensor.data_ptr(), ffn_inner);

                layernorme_tensor.free();
            }
            else
            {
                fc1_mul(input.data_ptr(), ffn_inner);
            }

            add_bias_act(ffn_inner);
            Buffer& output = MManager::get_instance().get_cache(desc_.batch_size_ * desc_.max_full_seq_len_ * desc_.hidden_units_, desc_.dtype_, desc_.options_,"emb_ffn");

            fc2_mul(ffn_inner, output);

            ffn_inner.free();

            add_input_bias_layernorm(output,input,pre_layernorm, add_redusial);

            auto res = torch::from_blob(output.data_ptr(), input.sizes(), input.strides(), desc_.options_);
            return std::move(res);
        }

        // layerNorm
        void FeedForwardNetwork::layer_norm(const torch::Tensor& input_tensor, Buffer& layernorm_tensor)
        {
            const int m = cur_batch_size_ * cur_seq_len_;
            int n = desc_.hidden_units_;
            RUN_KERNEL(layernorm,desc_.dtype_,input_tensor.data_ptr(),layernorm_weights_,layernorm_bias_,layernorm_tensor.data_ptr(), m, n, desc_.stream);
#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void FeedForwardNetwork::fc1_mul(void* input, Buffer &ffn_inner)
        {
            const int m = cur_batch_size_ * cur_seq_len_;
            int k = desc_.hidden_units_ ;
            int n = 4 * k;
            
            check_cuda_error(cublasGemmEx(desc_.cublasHandle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                alpha_,
                                intermediate_weights_, desc_.computeType_, n,
                                input, desc_.computeType_, k,
                                beta_,
                                ffn_inner.data_ptr(), desc_.computeType_, n,
                                desc_.computeType_,
                                fc1_algo_));

#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void FeedForwardNetwork::add_bias_act(Buffer& ffn_inner)
        {
            int m = cur_batch_size_ * cur_seq_len_;
            int n = 4 * desc_.hidden_units_ ;
            
            RUN_KERNEL(add_bias_act_kernel,desc_.dtype_,ffn_inner.data_ptr(), intermediate_bias_, m, n ,act_type_ ,desc_.stream)
            
#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void FeedForwardNetwork::fc2_mul(const Buffer& ffn_inner, Buffer& output)
        { 
            const int m = cur_batch_size_ * cur_seq_len_;
            int n = desc_.hidden_units_ ;
            int k = 4 * n;

            check_cuda_error(cublasGemmEx(desc_.cublasHandle,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          n, m, k,
                                          alpha_,
                                          output_weights_, desc_.computeType_, n,
                                          ffn_inner.data_ptr(), desc_.computeType_, k,
                                          beta_,
                                          output.data_ptr(), desc_.computeType_, n,
                                          desc_.computeType_,
                                          fc2_algo_));
            

#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void FeedForwardNetwork::add_input_bias_layernorm(Buffer& output,torch::Tensor& input,bool pre_layernorm, bool add_redusial)
        {
            const int m = cur_batch_size_ * cur_seq_len_;
            int n = desc_.hidden_units_ ;
            int k = 4 * n;

            if(add_redusial)
            {   
                if(!pre_layernorm)
                {   
                    // add_bias + add_redusial + layer_norm
                    RUN_KERNEL(add_bias_input_layernorm_kernel,desc_.dtype_,
                                        output.data_ptr(),input.data_ptr(), 
                                        output_bias_,layernorm_weights_,
                                        layernorm_bias_,m , n, desc_.stream);
                }
                else
                {
                    // add_bias + add_redusial
                    RUN_KERNEL(add_bias_input_kernel, desc_.dtype_, output.data_ptr(), input.data_ptr(),output_bias_, m , n, desc_.stream);
                }
            }
            else
            {
                // only add bias
                RUN_KERNEL(add_bias_kernel, desc_.dtype_, output.data_ptr(), output_bias_,m , n, desc_.stream);
            }
        }
    } // namespace op
} // namespace eet
