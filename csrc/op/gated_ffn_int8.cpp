#include "op/gated_ffn_int8.hpp"
#include "core/add_bias.cuh"
#include "core/layer_norm.cuh"
#include "core/gpt2_self_softmax.cuh"
#include "core/activation_kernel.cuh"
#include "cutlass_kernels/fpA_intB_gemm.h"

namespace eet
{
    namespace op
    {
        GatedFeedForwardNetworkInt8::GatedFeedForwardNetworkInt8(MetaDesc desc,
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
                            const std::string& ffn_cache_name) : 
        desc_(desc),
        intermediate_0_weights_(Intermediate_0_weights.data_ptr()),
        intermediate_0_scale_(Intermediate_0_scale.data_ptr()),
        intermediate_0_bias_(Intermediate_0_bias.data_ptr()),
        intermediate_1_weights_(Intermediate_1_weights.data_ptr()),
        intermediate_1_scale_(Intermediate_1_scale.data_ptr()),
        intermediate_1_bias_(Intermediate_1_bias.data_ptr()),
        output_weights_(Output_weights.data_ptr()),
        output_scale_(Output_scale.data_ptr()),
        output_bias_(Output_bias.data_ptr()),
        layernorm_weights_(layernorm_weights.data_ptr()),
        layernorm_bias_(layernorm_bias.data_ptr()),
        ffn_cache_name_(ffn_cache_name)
        {   
            // Currently only supports gelu and relu
            if (desc_.activation_fn_ == "silu")
            {
                act_type_ = 3;
            }
            else if (desc_.activation_fn_ == "quick_gelu")
            {
                act_type_ = 2;
            }
            else if (desc_.activation_fn_ == "gelu" || desc_.activation_fn_ == "gelu_new" || desc_.activation_fn_ == "gelu_fast")
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
            if (desc_.d_ff_ == 0) {
                size_per_head_ = desc_.hidden_units_ / desc_.head_num_;
                d_ff_ = desc_.hidden_units_ * 4;
            } else {
                size_per_head_ = desc_.d_kv_;
                d_ff_ = desc_.d_ff_;
            }
            MManager::get_instance().get_cache(desc_.batch_size_ * desc_.max_seq_len_ * desc_.hidden_units_, desc_.dtype_, desc_.options_, ffn_cache_name_);
            // MManager::get_instance().allocate_buffer(desc_.batch_size_ * desc_.max_seq_len_ * d_ff_, desc_.dtype_, desc_.options_, "gated_ffn_buffer1");
            // MManager::get_instance().allocate_buffer(desc_.batch_size_ * desc_.max_seq_len_ * d_ff_, desc_.dtype_, desc_.options_, "gated_ffn_buffer2");
            
            assert((desc_.dtype_ == torch::kFloat16) && "MaskedMultiHeadAttentionInt8 only support fp16");

            fc1_algo_ = CUBLAS_GEMM_DEFAULT;
            fc2_algo_ = CUBLAS_GEMM_DEFAULT;
            fc3_algo_ = CUBLAS_GEMM_DEFAULT;
            alpha_ = new half();
            beta_ = new half();
            *((half *)alpha_) = (half)1.0f;
            *((half *)beta_) = (half)0.0f;
        }

        torch::Tensor GatedFeedForwardNetworkInt8::forward(torch::Tensor &input,
                                                    bool pre_layernorm,
                                                    bool add_residual)
        {
            assert((input.dtype() == desc_.dtype_) && "input's dtype is not the same as GatedFeedForwardNetworkInt8's dtype");
            cur_batch_size_ = input.sizes()[0];
            cur_seq_len_ = input.sizes()[1];

            //ffn_inner
            Buffer &ffn_inner_act = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ * d_ff_, desc_.dtype_, desc_.options_, false, "ffn_inner_act");
            Buffer &ffn_inner_linear = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ * d_ff_, desc_.dtype_, desc_.options_, false, "ffn_inner_linear");

            // pre_layerNorm
            Buffer& layernorm_tensor = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ * desc_.hidden_units_, desc_.dtype_, desc_.options_, false, "layernorm_tensor");
            layer_norm(input, layernorm_tensor);

            fc1_mul_opt(layernorm_tensor.data_ptr(), ffn_inner_act);

            fc2_mul_opt(layernorm_tensor.data_ptr(), ffn_inner_linear);

            layernorm_tensor.free();
            gated_act(ffn_inner_act, ffn_inner_linear);
            ffn_inner_linear.free();

            Buffer &output = MManager::get_instance().get_cache(desc_.batch_size_ * desc_.max_seq_len_ * desc_.hidden_units_, desc_.dtype_, desc_.options_, ffn_cache_name_);

            fc3_mul_opt(ffn_inner_act, output);

            ffn_inner_act.free();

            add_input_bias_layernorm(output, input, pre_layernorm, add_residual);

            auto res = torch::from_blob(output.data_ptr(), input.sizes(), input.strides(), desc_.options_);
            return std::move(res);
        }

        // layerNorm
        void GatedFeedForwardNetworkInt8::layer_norm(const torch::Tensor& input_tensor, Buffer& layernorm_tensor)
        {
            const int m = cur_batch_size_ * cur_seq_len_;
            int n = desc_.hidden_units_;
            if (layernorm_bias_ != nullptr)
            {
                RUN_KERNEL(layernorm, desc_.dtype_, input_tensor.data_ptr(), layernorm_weights_, layernorm_bias_, layernorm_tensor.data_ptr(), m, n, desc_.layernorm_eps_, desc_.stream);
            }
            else
            {
                RUN_KERNEL(RMSnorm, desc_.dtype_, input_tensor.data_ptr(), layernorm_weights_, layernorm_tensor.data_ptr(), m, n, desc_.layernorm_eps_, desc_.stream);
            }

#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void GatedFeedForwardNetworkInt8::add_bias_act(Buffer& ffn_inner)
        {
            int m = cur_batch_size_ * cur_seq_len_;
            int n = d_ff_;
            
            RUN_KERNEL(add_bias_act_kernel,desc_.dtype_,ffn_inner.data_ptr(), intermediate_0_bias_, m, n, act_type_ ,desc_.stream)
            
#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void GatedFeedForwardNetworkInt8::gated_act(Buffer& inner_act, Buffer& inner_linear)
        {
            int m = cur_batch_size_ * cur_seq_len_;
            int n = d_ff_;
            
            RUN_KERNEL(gated_act_kernel, desc_.dtype_, inner_act.data_ptr(), inner_linear.data_ptr(), m, n, act_type_ ,desc_.stream)
            
#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void GatedFeedForwardNetworkInt8::fc1_mul(void* input, Buffer &ffn_inner, Buffer& weight_buf)
        {
            const int m = cur_batch_size_ * cur_seq_len_;
            int k = desc_.hidden_units_ ;
            int n = d_ff_;
            
            check_cuda_error(cublasGemmEx(desc_.cublasHandle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                alpha_,
                                weight_buf.data_ptr(), desc_.dataType_ , n,
                                input, desc_.dataType_ , k,
                                beta_,
                                ffn_inner.data_ptr(), desc_.dataType_ , n,
                                desc_.computeType_,
                                fc1_algo_));

#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void GatedFeedForwardNetworkInt8::fc2_mul(void* input, Buffer &ffn_inner,Buffer& weight_buf)
        {
            const int m = cur_batch_size_ * cur_seq_len_;
            int k = desc_.hidden_units_ ;
            int n = d_ff_;

            check_cuda_error(cublasGemmEx(desc_.cublasHandle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                alpha_,
                                weight_buf.data_ptr(), desc_.dataType_ , n,
                                input, desc_.dataType_ , k,
                                beta_,
                                ffn_inner.data_ptr(), desc_.dataType_ , n,
                                desc_.computeType_,
                                fc2_algo_));

#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void GatedFeedForwardNetworkInt8::fc3_mul(const Buffer& ffn_inner, Buffer& output,Buffer& weight_buf)
        { 
            const int m = cur_batch_size_ * cur_seq_len_;
            int n = desc_.hidden_units_ ;
            int k = d_ff_;

            check_cuda_error(cublasGemmEx(desc_.cublasHandle,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          n, m, k,
                                          alpha_,
                                          weight_buf.data_ptr(), desc_.dataType_ , n,
                                          ffn_inner.data_ptr(), desc_.dataType_ , k,
                                          beta_,
                                          output.data_ptr(), desc_.dataType_ , n,
                                          desc_.computeType_,
                                          fc3_algo_));
            

#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }


        void GatedFeedForwardNetworkInt8::fc1_mul_opt(void* input, Buffer &ffn_inner)
        {
            const int m = cur_batch_size_ * cur_seq_len_;
            int k = desc_.hidden_units_ ;
            int n = d_ff_;

            fastertransformer::gemm_fp16_int_bias_act(
                reinterpret_cast<fastertransformer::half *>(input),
                reinterpret_cast<const uint8_t *>(intermediate_0_weights_),
                reinterpret_cast<fastertransformer::half *>(intermediate_0_scale_),
                nullptr,
                reinterpret_cast<fastertransformer::half *>(ffn_inner.data_ptr()),
                std::nullopt,
                m, n, k,
                0,
                nullptr,
                0,
                desc_.stream);

#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void GatedFeedForwardNetworkInt8::fc2_mul_opt(void* input, Buffer &ffn_inner)
        {
            const int m = cur_batch_size_ * cur_seq_len_;
            int k = desc_.hidden_units_ ;
            int n = d_ff_;

            fastertransformer::gemm_fp16_int_bias_act(
                reinterpret_cast<fastertransformer::half *>(input),
                reinterpret_cast<const uint8_t *>(intermediate_1_weights_),
                reinterpret_cast<fastertransformer::half *>(intermediate_1_scale_),
                nullptr,
                reinterpret_cast<fastertransformer::half *>(ffn_inner.data_ptr()),
                std::nullopt,
                m, n, k,
                0,
                nullptr,
                0,
                desc_.stream);

#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void GatedFeedForwardNetworkInt8::fc3_mul_opt(const Buffer& ffn_inner, Buffer& output)
        { 
            const int m = cur_batch_size_ * cur_seq_len_;
            int n = desc_.hidden_units_ ;
            int k = d_ff_;

            fastertransformer::gemm_fp16_int_bias_act(
                reinterpret_cast<fastertransformer::half *>(ffn_inner.data_ptr()),
                reinterpret_cast<const uint8_t *>(output_weights_),
                reinterpret_cast<fastertransformer::half *>(output_scale_),
                nullptr,
                reinterpret_cast<fastertransformer::half *>(output.data_ptr()),
                std::nullopt,
                m, n, k,
                0,
                nullptr,
                0,
                desc_.stream);

#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void GatedFeedForwardNetworkInt8::add_input_bias_layernorm(Buffer& output,torch::Tensor& input,bool pre_layernorm, bool add_residual)
        {
            const int m = cur_batch_size_ * cur_seq_len_;
            int n = desc_.hidden_units_ ;
            int k = d_ff_;

            if(add_residual)
            {   
                if(!pre_layernorm)
                {   
                    // add_bias + add_residual + layer_norm
                    RUN_KERNEL(add_bias_input_layernorm_kernel,desc_.dtype_,
                                        output.data_ptr(),input.data_ptr(), 
                                        output_bias_,layernorm_weights_,
                                        layernorm_bias_, m , n, desc_.layernorm_eps_, desc_.stream);
                }
                else
                {
                    // add_bias + add_residual
                    RUN_KERNEL(add_bias_input_kernel, desc_.dtype_, output.data_ptr(), input.data_ptr(),output_bias_, m , n, desc_.stream);
                }
            }
            else
            {
                // only add bias
                if (output_bias_ != nullptr) {
                    RUN_KERNEL(add_bias_kernel, desc_.dtype_, output.data_ptr(), output_bias_,m , n, desc_.stream);
                }
            }
        }
    } // namespace op
} // namespace eet
