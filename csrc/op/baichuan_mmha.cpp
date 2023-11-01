#include "op/baichuan_mmha.hpp"
#include "core/add_bias.cuh"
#include "core/transpose.cuh"
#include "core/layer_norm.cuh"
#include "core/pre_process.cuh"
#include "core/self_add_bias.cuh"
#include "core/attention_dispatch.cuh"
#include "core/gpt2_self_softmax.cuh"
#include "cutlass_kernels/fpA_intB_gemm.h"

namespace eet{
    namespace op{
        BaichuanMmha::BaichuanMmha(MetaDesc desc,
                                    const torch::Tensor& QKV_weights,
                                    const torch::Tensor& QKV_scale,
                                    const torch::Tensor& Q_bias,
                                    const torch::Tensor& K_bias,
                                    const torch::Tensor& V_bias,
                                    const torch::Tensor& Output_weights,
                                    const torch::Tensor& Output_scale,
                                    const torch::Tensor& Output_bias,
                                    const torch::Tensor& layernorm_weights,
                                    const torch::Tensor& layernorm_bias):
            desc_(desc),
            step_(1),
            qkv_weights_(QKV_weights.data_ptr()),
            qkv_scale_(QKV_scale.data_ptr()),
            q_bias_(Q_bias.data_ptr()),
            k_bias_(K_bias.data_ptr()),
            v_bias_(V_bias.data_ptr()),
            output_weights_(Output_weights.data_ptr()),
            output_scale_(Output_scale.data_ptr()),
            output_bias_(Output_bias.data_ptr()),
            layernorm_weights_(layernorm_weights.data_ptr()),
            layernorm_bias_(layernorm_bias.data_ptr())
        {
            if (desc_.d_kv_ == 0) {
                size_per_head_ = desc_.hidden_units_ / desc_.head_num_;
                inner_dim_ = desc_.hidden_units_;
            } else {
                size_per_head_ = desc_.d_kv_;
                inner_dim_ = size_per_head_ * desc_.head_num_;
            }

            with_bias_ = q_bias_ != nullptr ? true : false;
            k_cache_ = torch::zeros({desc_.batch_size_ * desc_.max_full_seq_len_ * inner_dim_}, desc_.options_);          // each layer has kv cache
            v_cache_ = torch::zeros_like(k_cache_);
            MManager::get_instance().get_cache(desc_.batch_size_ * desc_.max_seq_len_ * desc_.hidden_units_, desc_.dtype_, desc_.options_, "self_mask_attn_cache");

            assert((desc_.dtype_ == torch::kFloat16) && "BaichuanMmha only support fp16");

            qkv_weights_algo_ = CUBLAS_GEMM_DEFAULT;
            q_k_algo_ = CUBLAS_GEMM_DEFAULT;
            attn_v_algo_ = CUBLAS_GEMM_DEFAULT;
            alpha_ = new half();
            beta_ = new half();
            atten_scaler_ = new half();
            *((half *)alpha_) = 1.0f;
            *((half *)beta_) = 0.0f;
            *((half *)atten_scaler_) = sqrt(1.0f / size_per_head_);     // TODO: T5 is different
        }

        torch::Tensor BaichuanMmha::forward(torch::Tensor& input,
                                    const torch::Tensor& pre_padding_len,
                                    const torch::Tensor& reorder_state,
                                    bool pre_layernorm,
                                    bool add_residual,
                                    bool first_pass,
                                    const torch::Tensor &relative_attention_bias){
            if(first_pass)
            {
                return forward_full(input, pre_padding_len, pre_layernorm, add_residual, relative_attention_bias);
            }
            else
            {
                return forward_inc(input, pre_padding_len, reorder_state, pre_layernorm, add_residual, relative_attention_bias);
            }
        }

        // full decoder
        torch::Tensor BaichuanMmha::forward_full(torch::Tensor &input,
                                                             const torch::Tensor &pre_padding_length,
                                                             bool pre_layernorm,
                                                             bool add_residual,
                                                             const torch::Tensor &relative_attention_bias)
        {
            assert((input.dtype() == desc_.dtype_) && "input's dtype is not the same as BaichuanMmha's dtype");
            step_ = 1;
            cur_batch_size_ = input.sizes()[0];
            first_batch_size_ = cur_batch_size_;
            cur_seq_len_ = input.sizes()[1];
            assert((cur_seq_len_ <= desc_.max_full_seq_len_)&& "cur_seq_len must be less than or equal to max_full_seq_len_");
            assert((cur_batch_size_ <= desc_.batch_size_)&& "cur_batch_size_ must be less than or equal to max_batch_size_");

            Buffer& qkv_buffer = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ * inner_dim_ * 3, desc_.dtype_, desc_.options_, false, "qkv_full");
            

            // pre_layerNorm
            Buffer& layernormed_query = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ * 
                                                    desc_.hidden_units_, desc_.dtype_, desc_.options_, false, "layernorm");
            layer_norm(input, layernormed_query);
            // fused gemm with dequant
            if (desc_.is_int8_) {
                qkv_weights_mul_weight_only_int8(layernormed_query.data_ptr(), qkv_buffer);
            } else {
                qkv_weights_mul(layernormed_query.data_ptr(), qkv_buffer);
            }
            layernormed_query.free();

            // qkv transpose
            Buffer& q_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_, false, "q_buf");
            Buffer& k_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_, false, "k_buf");
            Buffer& v_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_, false, "v_buf");
            qkv_add_bias(qkv_buffer, q_buf, k_buf, v_buf);
            qkv_buffer.free();

            // q * k
            Buffer& qk_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.head_num_ *
                                        desc_.max_seq_len_ * desc_.max_seq_len_, desc_.dtype_, desc_.options_, false, "qk_buf");
            q_k_mul(q_buf, k_buf, qk_buf);

            q_buf.free();

            // relative attention bias
            void* relative_attention_bias_ = relative_attention_bias.data_ptr();

            // softmax
            const int64_t *padding_len = pre_padding_length.data_ptr<int64_t>();
            qk_softmax(qk_buf, relative_attention_bias_, padding_len);

            // transpose k\v cache
            kv_transpose(k_cache_, v_cache_, k_buf, v_buf);
            k_buf.free();

            // attn * v
            Buffer& transpose_dst = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_, false, "transpose_dst");
            attn_v_mul(qk_buf, v_buf, transpose_dst);
            qk_buf.free();
            v_buf.free();

            // transpose
            Buffer& dst = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_, false, "dst");

            transpose(transpose_dst, dst);
            transpose_dst.free();

            // project
            Buffer &output = MManager::get_instance().get_cache(desc_.batch_size_ * desc_.max_seq_len_ * 
                                       desc_.hidden_units_, desc_.dtype_, desc_.options_, "self_mask_attn_cache");

            project(dst, output, input, pre_layernorm, add_residual);

            dst.free();
            step_ = cur_seq_len_;

            auto res = torch::from_blob(output.data_ptr(), input.sizes(), input.strides(), desc_.options_);
            return std::move(res);
        }

        // incremental decoder
        torch::Tensor BaichuanMmha::forward_inc(torch::Tensor &input,
                                                            const torch::Tensor &pre_padding_len,
                                                            const torch::Tensor &reorder_state,
                                                            bool pre_layernorm,
                                                            bool add_residual,
                                                            const torch::Tensor &relative_attention_bias)
        {
            assert((input.dtype() == desc_.dtype_) && "input's dtype is not the same as BaichuanMmha's dtype");
            step_ += 1;
            assert(step_ <= desc_.max_full_seq_len_ && "Exceed the maximum step length");
            cur_batch_size_ = input.sizes()[0];
            cur_seq_len_ = input.sizes()[1];
            assert(cur_seq_len_ == 1);
            Buffer& qkv_buffer = MManager::get_instance().get_buffer(desc_.batch_size_ * cur_seq_len_ *
                                    inner_dim_ * 3, desc_.dtype_, desc_.options_, false, "qkv_inc");


            // pre_layerNorm
            Buffer& layernormed_query = MManager::get_instance().get_buffer(desc_.batch_size_ * cur_seq_len_ *
                    desc_.hidden_units_, desc_.dtype_, desc_.options_, false, "layernorm_inc");
            layer_norm(input, layernormed_query);
            if (desc_.is_int8_) {
                qkv_weights_mul_weight_only_int8(layernormed_query.data_ptr(), qkv_buffer);
            } else {
                qkv_weights_mul(layernormed_query.data_ptr(), qkv_buffer);
            }
            layernormed_query.free();

            // TODO rotary
            qkv_buffer.free();
            // masked_attention_dispatch
            Buffer &context_buf = MManager::get_instance().get_buffer(
                desc_.batch_size_ * cur_seq_len_ * inner_dim_, desc_.dtype_, desc_.options_, false, "context_buf");
            const int64_t *padding_len = pre_padding_len.data_ptr<int64_t>();
            const int64_t *reorder_index = reorder_state.data_ptr<int64_t>();
            void* relative_attention_bias_ = relative_attention_bias.data_ptr();

            if (reorder_index != nullptr) {
                reorder_cache(k_cache_, v_cache_, k_cache_, v_cache_, reorder_index);     //TODO 不能保证读写读顺序
            }

            fused_masked_attention(qkv_buffer, context_buf, padding_len, nullptr, relative_attention_bias_);
            
            // attention output
            Buffer& output = MManager::get_instance().get_cache(desc_.batch_size_ * cur_seq_len_ * desc_.hidden_units_, desc_.dtype_, desc_.options_, "self_mask_attn_cache");

            project(context_buf, output, input, pre_layernorm, add_residual);
            context_buf.free();
            auto res = torch::from_blob(output.data_ptr(), input.sizes(), input.strides(), desc_.options_);
            return std::move(res);
        }

        // layerNorm
        void BaichuanMmha::layer_norm(const torch::Tensor& input_tensor, Buffer& layernorm_query)
        {
            const int m = cur_batch_size_ * cur_seq_len_;
            int n = desc_.hidden_units_;

            if (with_bias_)
            {
                RUN_KERNEL(layernorm, desc_.dtype_, input_tensor.data_ptr(), layernorm_weights_, layernorm_bias_, layernorm_query.data_ptr(), m, n, desc_.layernorm_eps_, desc_.stream);
            }
            else
            {
                RUN_KERNEL(RMSnorm, desc_.dtype_, input_tensor.data_ptr(), layernorm_weights_, layernorm_query.data_ptr(), m, n, desc_.layernorm_eps_, desc_.stream);
            }
#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }


        void BaichuanMmha::qkv_weights_mul(void* input, 
                                    Buffer& qkv_buffer){
                const int m = cur_batch_size_ * cur_seq_len_;
                const int k = desc_.hidden_units_;
                const int n = 3 * inner_dim_;

                check_cuda_error(cublasGemmEx(desc_.cublasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    alpha_,
                    qkv_weights_, desc_.dataType_ , n,
                    input, desc_.dataType_ , k,
                    beta_,
                    qkv_buffer.data_ptr(), desc_.dataType_ , n,
                    desc_.computeType_,
                    qkv_weights_algo_));


#ifdef _DEBUG_MODE_
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
        }

        void BaichuanMmha::qkv_weights_mul_weight_only_int8(void *input,
                                                                        Buffer &qkv_buffer)
        {
            int m = cur_batch_size_ * cur_seq_len_;
            int k = desc_.hidden_units_;
            int n = 3 * inner_dim_;

            fastertransformer::gemm_fp16_int_bias_act(
                reinterpret_cast<fastertransformer::half *>(input),
                reinterpret_cast<const uint8_t*>(qkv_weights_),
                reinterpret_cast<fastertransformer::half *>(qkv_scale_),
                nullptr,
                reinterpret_cast<fastertransformer::half *>(qkv_buffer.data_ptr()),
                std::nullopt,
                m, n, k,
                0,
                nullptr,
                0,
                desc_.stream
                );

#ifdef _DEBUG_MODE_
                cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
        }

        void BaichuanMmha::qkv_add_bias(const Buffer &qkv_buffer,
                                                       Buffer &q_buf,
                                                       Buffer &k_buf,
                                                       Buffer &v_buf)
        {
            RUN_KERNEL(fused_add_QKV_bias_kernel,desc_.dtype_,qkv_buffer.data_ptr(),q_bias_,
                       k_bias_, v_bias_, q_buf.data_ptr(), k_buf.data_ptr(), v_buf.data_ptr(),
                       cur_batch_size_, cur_seq_len_, desc_.head_num_, size_per_head_, desc_.stream);
#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }


        void BaichuanMmha::q_k_mul(const Buffer& q_buf, const Buffer& k_buf,
                                                Buffer& qk_buf){
            check_cuda_error(cublasGemmStridedBatchedEx(desc_.cublasHandle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                cur_seq_len_, cur_seq_len_, size_per_head_,
                atten_scaler_,
                k_buf.data_ptr(), desc_.dataType_ , size_per_head_, cur_seq_len_ * size_per_head_,
                q_buf.data_ptr(), desc_.dataType_ , size_per_head_, cur_seq_len_ * size_per_head_,
                beta_,
                qk_buf.data_ptr(), desc_.dataType_ , cur_seq_len_, cur_seq_len_ * cur_seq_len_,
                cur_batch_size_ * desc_.head_num_,
                desc_.computeType_,
                q_k_algo_));

#ifdef _DEBUG_MODE_
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif
        }

        void BaichuanMmha::qk_softmax(Buffer& qk_buf, void* relative_attention_bias, const int64_t *padding_len){
            // float scalar = 1 / sqrtf(size_per_head_ * 1.0f);

            RUN_KERNEL(launch_masked_softmax_kernel, desc_.dtype_, qk_buf.data_ptr(), relative_attention_bias, padding_len, cur_batch_size_,
                    desc_.head_num_, cur_seq_len_, desc_.stream);
#ifdef _DEBUG_MODE_
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif 
        }

        void BaichuanMmha::attn_v_mul(const Buffer& qk_buf,
                                             const Buffer& v_buf,
                                             Buffer& transpose_dst){
            check_cuda_error(cublasGemmStridedBatchedEx(desc_.cublasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    size_per_head_, cur_seq_len_, cur_seq_len_,
                    alpha_,
                    v_buf.data_ptr(), desc_.dataType_ , size_per_head_, cur_seq_len_ * size_per_head_,
                    qk_buf.data_ptr(), desc_.dataType_ , cur_seq_len_, cur_seq_len_ * cur_seq_len_,
                    beta_,
                    transpose_dst.data_ptr(), desc_.dataType_ , size_per_head_, cur_seq_len_ * size_per_head_,
                    cur_batch_size_ * desc_.head_num_,
                    desc_.computeType_,
                    attn_v_algo_));

#ifdef _DEBUG_MODE_
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif
        }

        void BaichuanMmha::transpose(const Buffer& transpose_dst, Buffer&  dst){
            RUN_KERNEL(transpose_kernel,desc_.dtype_,transpose_dst.data_ptr(),dst.data_ptr(), cur_batch_size_, cur_seq_len_,
                    desc_.head_num_, size_per_head_, desc_.stream);

            #ifdef _DEBUG_MODE_
                cudaDeviceSynchronize();
                check_cuda_error(cudaGetLastError());
            #endif
        }


        void BaichuanMmha::project(const Buffer &dst, Buffer &res, torch::Tensor &input, bool pre_layernorm, bool add_residual)
        {
            const int m = cur_batch_size_ * cur_seq_len_;
            int k = inner_dim_;
            int n = desc_.hidden_units_;

            if (desc_.is_int8_)
            {
                fastertransformer::gemm_fp16_int_bias_act(
                    reinterpret_cast<fastertransformer::half *>(dst.data_ptr()),
                    reinterpret_cast<const uint8_t *>(output_weights_),
                    reinterpret_cast<fastertransformer::half *>(output_scale_),
                    nullptr,
                    reinterpret_cast<fastertransformer::half *>(res.data_ptr()),
                    std::nullopt,
                    m, n, k,
                    0,
                    nullptr,
                    0,
                    desc_.stream);
            } else {
                check_cuda_error(cublasGemmEx(desc_.cublasHandle,
                                              CUBLAS_OP_N, CUBLAS_OP_N,
                                              n, m, k,
                                              alpha_,
                                              output_weights_, desc_.dataType_, n,
                                              dst.data_ptr(), desc_.dataType_, k,
                                              beta_,
                                              res.data_ptr(), desc_.dataType_, n,
                                              desc_.computeType_,
                                              qkv_weights_algo_));
            }

                // add_bias + add_residual
                RUN_KERNEL(add_bias_input_kernel, desc_.dtype_, res.data_ptr(), input.data_ptr(), output_bias_,
                           m, n, desc_.stream);

            #ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
            #endif
            }

        void BaichuanMmha::kv_transpose(torch::Tensor& d_K_buf, torch::Tensor& d_V_buf,Buffer& K_buf,Buffer& V_buf)
         {
            RUN_KERNEL(copyKV_transpose_kernel,desc_.dtype_,d_K_buf.data_ptr(), d_V_buf.data_ptr(),K_buf.data_ptr(), V_buf.data_ptr(),cur_batch_size_, cur_seq_len_,
                   desc_.head_num_, size_per_head_);
            #ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
            #endif
         }

        void BaichuanMmha::fused_masked_attention(const Buffer &qkv_buffer,
                                                               Buffer &context_buf,
                                                               const int64_t *padding_len,
                                                               const int64_t *reorder_index,
                                                               const void *relative_attention_bias_)
         {
            RUN_KERNEL(fused_masked_attention_dispatch,desc_.dtype_,qkv_buffer.data_ptr(), q_bias_,k_cache_.data_ptr(),k_bias_,v_cache_.data_ptr(),v_bias_,
                        context_buf.data_ptr(),cur_batch_size_,first_batch_size_,desc_.head_num_,size_per_head_,step_, desc_.stream, padding_len,reorder_index, relative_attention_bias_);
            #ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
            #endif
         }

        void BaichuanMmha::reorder_cache(torch::Tensor &K_cache, torch::Tensor &V_cache, Buffer &K_buf, Buffer &V_buf, const int64_t *reorder_index)
         {
            RUN_KERNEL(reorderKV_kernel, desc_.dtype_, K_cache.data_ptr(), V_cache.data_ptr(), K_buf.data_ptr(), V_buf.data_ptr(), reorder_index, first_batch_size_, step_, desc_.head_num_, size_per_head_);
#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
         }

        void BaichuanMmha::reorder_cache(torch::Tensor &K_cache, torch::Tensor &V_cache, torch::Tensor &K_buf, torch::Tensor &V_buf, const int64_t *reorder_index)
         {
            RUN_KERNEL(reorderKV_kernel, desc_.dtype_, K_cache.data_ptr(), V_cache.data_ptr(), K_buf.data_ptr(), V_buf.data_ptr(), reorder_index, first_batch_size_, step_, desc_.head_num_, size_per_head_);
#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
         }


        void BaichuanMmha::masked_attention(const Buffer &q_buf,
                                                        const Buffer &k_buf,
                                                        const Buffer &v_buf,
                                                        Buffer &context_buf,
                                                        const int64_t *padding_len,
                                                        const int64_t *reorder_index)
         {
            RUN_KERNEL(masked_attention_dispatch, desc_.dtype_, k_buf.data_ptr(), v_buf.data_ptr(), 
                       q_buf.data_ptr(), k_cache_.data_ptr(), v_cache_.data_ptr(),
                       context_buf.data_ptr(), cur_batch_size_, first_batch_size_, desc_.head_num_, size_per_head_, step_, desc_.stream, padding_len, reorder_index);
#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
         }
    }
}
