#include "op/masked_multi_head_attention.hpp"
#include "core/add_bias.cuh"
#include "core/transpose.cuh"
#include "core/layer_norm.cuh"
#include "core/pre_process.cuh"
#include "core/self_add_bias.cuh"
#include "core/attention_dispatch.cuh"
#include "core/gpt2_self_softmax.cuh"

// for gpt
namespace eet{
    namespace op{
        MaskedMultiHeadAttention::MaskedMultiHeadAttention(MetaDesc desc,
                                    const torch::Tensor& QKV_weights,
                                    const torch::Tensor& Q_bias,
                                    const torch::Tensor& K_bias,
                                    const torch::Tensor& V_bias,
                                    const torch::Tensor& Output_weights,
                                    const torch::Tensor& Output_bias,
                                    const torch::Tensor& layernorm_weights,
                                    const torch::Tensor& layernorm_bias):
            desc_(desc),
            step_(1),
            qkv_weights_(QKV_weights.data_ptr()),
            q_bias_(Q_bias.data_ptr()),
            k_bias_(K_bias.data_ptr()),
            v_bias_(V_bias.data_ptr()),
            output_weights_(Output_weights.data_ptr()),
            output_bias_(Output_bias.data_ptr()),
            layernorm_weights_(layernorm_weights.data_ptr()),
            layernorm_bias_(layernorm_bias.data_ptr())
        {
            // size_per_head_ = 64;
            size_per_head_ = desc_.hidden_units_ / desc_.head_num_;
            inner_dim_ = size_per_head_* desc_.head_num_;

            with_bias_ = q_bias_ != nullptr ? true : false;
            k_cache_ = torch::zeros({desc_.batch_size_, desc_.max_seq_len_, desc_.hidden_units_}, desc_.options_);
            v_cache_ = torch::zeros_like(k_cache_);
            check_cuda_error(cudaMalloc(&fused_qkv_ptr_,sizeof(void**) * FUSED_QKV_PTR_SIZE));
            Buffer& attn_out = MManager::get_instance().get_cache(desc_.batch_size_ * desc_.max_full_seq_len_ * desc_.hidden_units_, desc_.dtype_, desc_.options_,"attn_cache");

            qkv_kernel_ = (void**)fused_qkv_ptr_;
            qkv_input_  = qkv_kernel_ + QKV_PTR_SIZE;
            qkv_buf_   = qkv_input_  + QKV_PTR_SIZE;
            switch (desc_.dtype_)
            {
            case torch::kFloat32:
                qkv_weights_algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
                q_k_algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
                attn_v_algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
                alpha_ = new float();
                beta_ = new float();
                atten_scaler_ = new float();
                *((float *)alpha_) = 1.0f;
                *((float *)beta_) = 0.0f;
                if (with_bias_) {
                    *((float *)atten_scaler_) = sqrt(1.0f / size_per_head_);
                } else {
                    *((float *)atten_scaler_) = 1.0f;
                }                
                break;
            case torch::kFloat16:
                qkv_weights_algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
                q_k_algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
                attn_v_algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
                alpha_ = new half();
                beta_ = new half();
                atten_scaler_ = new half();
                *((half *)alpha_) = 1.0f;
                *((half *)beta_) = 0.0f;
                if (with_bias_) {
                    *((half *)atten_scaler_) = sqrt(1.0f / size_per_head_);
                } else {
                    *((half *)atten_scaler_) = 1.0f;
                }
                break;
            //TODO
            case torch::kInt8:
                break;
            }
        }

        torch::Tensor MaskedMultiHeadAttention::forward(torch::Tensor& input,
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
        torch::Tensor MaskedMultiHeadAttention::forward_full(torch::Tensor &input,
                                                             const torch::Tensor &pre_padding_length,
                                                             bool pre_layernorm,
                                                             bool add_residual,
                                                             const torch::Tensor &relative_attention_bias)
        {
            assert((input.dtype() == desc_.dtype_) && "input's dtype is not the same as MaskedMultiHeadAttention's dtype");
            step_ = 1;
            cur_batch_size_ = input.sizes()[0];
            first_batch_size_ = cur_batch_size_;
            cur_seq_len_ = input.sizes()[1];
            assert((cur_seq_len_ <= desc_.max_full_seq_len_)&& "cur_seq_len must be less than or equal to max_full_seq_len_");
            assert((cur_batch_size_ <= desc_.batch_size_)&& "cur_batch_size_ must be less than or equal to max_batch_size_");

            Buffer& qkv_buffer = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                                    desc_.hidden_units_ * 3, desc_.dtype_, desc_.options_, "qkv_buffer_full");
            if(pre_layernorm)
            {
                // pre_layerNorm
                Buffer& layernormed_query = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                        desc_.hidden_units_, desc_.dtype_, desc_.options_, "layernorm_query_full");
                layer_norm(input,layernormed_query);

                // qkv * weights
                qkv_weights_mul(layernormed_query.data_ptr(), qkv_buffer);

                layernormed_query.free();
            }
            else{
                // qkv * weights
                qkv_weights_mul(input.data_ptr(), qkv_buffer);
            }  

            // qkv add bias                
            Buffer& q_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_, "q_buf_full");
            Buffer& k_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_, "k_buf_full");
            Buffer& v_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_, "v_buf_full");
            qkv_add_bias(qkv_buffer, q_buf, k_buf, v_buf);

            qkv_buffer.free();


            // q * k
            Buffer& qk_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.head_num_ *
                                        desc_.max_full_seq_len_ * desc_.max_full_seq_len_, desc_.dtype_, desc_.options_, "qk_buf_full");
            q_k_mul(q_buf, k_buf, qk_buf);

            q_buf.free();

            // relative attention bias
            void* relative_attention_bias_ = relative_attention_bias.data_ptr();
            if (relative_attention_bias_ != nullptr) {
                add_relative_attn_bias(qk_buf, relative_attention_bias_);
            }

            // softmax
            const int64_t *padding_len = pre_padding_length.data_ptr<int64_t>();
            qk_softmax(qk_buf,padding_len);

            // attn * v
            Buffer& transpose_dst = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_, "transpose_dst_full");

            attn_v_mul(qk_buf,v_buf,transpose_dst);

            qk_buf.free();

            // transpose k\v cache
            kv_transpose(k_cache_,v_cache_,k_buf,v_buf);

            k_buf.free();
            v_buf.free();

            // transpose
            Buffer& dst = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_, "dst_full");

            transpose(transpose_dst, dst);
            transpose_dst.free();

            // project
            Buffer& output = MManager::get_instance().get_cache(desc_.batch_size_ * desc_.max_full_seq_len_ * desc_.hidden_units_, desc_.dtype_, desc_.options_,"attn_cache");

            project(dst,output,input ,pre_layernorm,add_residual);

            dst.free();
            step_ = cur_seq_len_;
            // output_ = output_[input.sizes()];
            auto res = torch::from_blob(output.data_ptr(), input.sizes(), input.strides(), desc_.options_);
            return std::move(res);
        }

        // incremental decoder
        torch::Tensor MaskedMultiHeadAttention::forward_inc(torch::Tensor &input,
                                                            const torch::Tensor &pre_padding_len,
                                                            const torch::Tensor &reorder_state,
                                                            bool pre_layernorm,
                                                            bool add_residual,
                                                            const torch::Tensor &relative_attention_bias)
        {
            assert((input.dtype() == desc_.dtype_) && "input's dtype is not the same as MaskedMultiHeadAttention's dtype");
            step_ += 1;
            assert(step_ <= desc_.max_seq_len_ && "Exceed the maximum step length");
            cur_batch_size_ = input.sizes()[0];
            cur_seq_len_ = input.sizes()[1];
            assert(cur_seq_len_ == 1);
            Buffer& qkv_buffer = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                                    desc_.hidden_units_ * 3, desc_.dtype_, desc_.options_, "qkv_buffer_inc");
            if(pre_layernorm)
            {
                // pre_layerNorm
                Buffer& layernormed_query = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                        desc_.hidden_units_, desc_.dtype_, desc_.options_, "layernorm_query_inc");
                layer_norm(input,layernormed_query);

                // qkv * weights
                qkv_weights_mul(layernormed_query.data_ptr(),qkv_buffer);
                layernormed_query.free();
            }
            else{
                // qkv * weights
                qkv_weights_mul(input.data_ptr(),qkv_buffer);
            }

            Buffer& context_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_full_seq_len_ *
                                    desc_.hidden_units_, desc_.dtype_, desc_.options_, "context_inc");

            // masked_attention_dispatch
            const int64_t *padding_len = pre_padding_len.data_ptr<int64_t>();
            const int64_t *reorder_index = reorder_state.data_ptr<int64_t>();
            void* relative_attention_bias_ = relative_attention_bias.data_ptr();
            
            masked_attention(qkv_buffer,context_buf,padding_len,reorder_index, relative_attention_bias_);
            qkv_buffer.free();
			
            Buffer& output = MManager::get_instance().get_cache(desc_.batch_size_ * desc_.max_full_seq_len_ * desc_.hidden_units_, desc_.dtype_, desc_.options_,"attn_cache");
            project(context_buf, output, input,pre_layernorm,add_residual);
            context_buf.free();
            auto res = torch::from_blob(output.data_ptr(), input.sizes(), input.strides(), desc_.options_);
            return std::move(res);
        }

        // layerNorm
        void MaskedMultiHeadAttention::layer_norm(const torch::Tensor& input_tensor, Buffer& layernorm_query)
        {
            const int m = cur_batch_size_ * cur_seq_len_;
            int n = desc_.hidden_units_;

            if (with_bias_) {
                RUN_KERNEL(layernorm,desc_.dtype_,input_tensor.data_ptr(),layernorm_weights_,layernorm_bias_,layernorm_query.data_ptr(), m, n, desc_.stream);
            } else {
                RUN_KERNEL(T5layernorm,desc_.dtype_,input_tensor.data_ptr(),layernorm_weights_,layernorm_query.data_ptr(), m, n, desc_.stream);
            }
#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void MaskedMultiHeadAttention::qkv_weights_mul(void* input, 
                                    Buffer& qkv_buffer){
                const int m = cur_batch_size_ * cur_seq_len_;
                const int k = desc_.hidden_units_;
                const int n = 3 * inner_dim_;

                check_cuda_error(cublasGemmEx(desc_.cublasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    alpha_,
                    qkv_weights_, desc_.computeType_, n,
                    input, desc_.computeType_, k,
                    beta_,
                    qkv_buffer.data_ptr(), desc_.computeType_, n,
                    desc_.computeType_,
                    qkv_weights_algo_));


#ifdef _DEBUG_MODE_
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
        }

        void MaskedMultiHeadAttention::qkv_add_bias(const Buffer &qkv_buffer,
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

        void MaskedMultiHeadAttention::q_k_mul(const Buffer& q_buf, const Buffer& k_buf,
                                                Buffer& qk_buf){
            check_cuda_error(cublasGemmStridedBatchedEx(desc_.cublasHandle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                cur_seq_len_, cur_seq_len_, size_per_head_,
                atten_scaler_,
                k_buf.data_ptr(), desc_.computeType_, size_per_head_, cur_seq_len_ * size_per_head_,
                q_buf.data_ptr(), desc_.computeType_, size_per_head_, cur_seq_len_ * size_per_head_,
                beta_,
                qk_buf.data_ptr(), desc_.computeType_, cur_seq_len_, cur_seq_len_ * cur_seq_len_,
                cur_batch_size_ * desc_.head_num_,
                desc_.computeType_,
                q_k_algo_));

#ifdef _DEBUG_MODE_
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif
        }

        void MaskedMultiHeadAttention::add_relative_attn_bias(Buffer &qk_buf, void *relative_attention_bias)
        {
            RUN_KERNEL(add_relative_attn_bias_kernel, desc_.dtype_, qk_buf.data_ptr(), relative_attention_bias,
                       cur_batch_size_, desc_.head_num_, cur_seq_len_, desc_.stream);

#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void MaskedMultiHeadAttention::qk_softmax(Buffer& qk_buf,const int64_t *padding_len){
            // float scalar = 1 / sqrtf(size_per_head_ * 1.0f);

            RUN_KERNEL(softmax_kernel,desc_.dtype_,qk_buf.data_ptr(), padding_len,  cur_batch_size_,
                    desc_.head_num_,cur_seq_len_, desc_.stream);
#ifdef _DEBUG_MODE_
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif 
        }

        void MaskedMultiHeadAttention::attn_v_mul(const Buffer& qk_buf,
                                             const Buffer& v_buf,
                                             Buffer& transpose_dst){
            check_cuda_error(cublasGemmStridedBatchedEx(desc_.cublasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    size_per_head_, cur_seq_len_, cur_seq_len_,
                    alpha_,
                    v_buf.data_ptr(), desc_.computeType_, size_per_head_, cur_seq_len_ * size_per_head_,
                    qk_buf.data_ptr(), desc_.computeType_, cur_seq_len_, cur_seq_len_ * cur_seq_len_,
                    beta_,
                    transpose_dst.data_ptr(), desc_.computeType_, size_per_head_, cur_seq_len_ * size_per_head_,
                    cur_batch_size_ * desc_.head_num_,
                    desc_.computeType_,
                    attn_v_algo_));

#ifdef _DEBUG_MODE_
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif
        }

        void MaskedMultiHeadAttention::transpose(const Buffer& transpose_dst, Buffer&  dst){
            RUN_KERNEL(transpose_kernel,desc_.dtype_,transpose_dst.data_ptr(),dst.data_ptr(), cur_batch_size_, cur_seq_len_,
                    desc_.head_num_, size_per_head_, desc_.stream);

            #ifdef _DEBUG_MODE_
                cudaDeviceSynchronize();
                check_cuda_error(cudaGetLastError());
            #endif
        }

        void MaskedMultiHeadAttention::project(const Buffer& dst, Buffer& res,torch::Tensor& input, bool pre_layernorm,bool add_residual){
            const int m = cur_batch_size_ * cur_seq_len_;
            int k = inner_dim_;
            int n = desc_.hidden_units_;
            check_cuda_error(cublasGemmEx(desc_.cublasHandle,
                                            CUBLAS_OP_N, CUBLAS_OP_N,
                                            n, m, k,
                                            alpha_,
                                            output_weights_, desc_.computeType_, n,
                                            dst.data_ptr(), desc_.computeType_, k,
                                            beta_,
                                            res.data_ptr(), desc_.computeType_, n,
                                            desc_.computeType_,
                                            qkv_weights_algo_));
            if(add_residual)
            {   
                if(!pre_layernorm)
                {   
                    // add_bias + add_residual + layer_norm
                    RUN_KERNEL(add_bias_input_layernorm_kernel,desc_.dtype_,
                                        res.data_ptr(),input.data_ptr(), 
                                        output_bias_,layernorm_weights_,
                                        layernorm_bias_,m , n, desc_.stream);
                }
                else
                {
                    // add_bias + add_residual
                    RUN_KERNEL(add_bias_input_kernel, desc_.dtype_, res.data_ptr(), input.data_ptr(),output_bias_,
                           m , n, desc_.stream);
                }
            }
            else
            {
                if (with_bias_) {
                    // only add bias
                    RUN_KERNEL(add_bias_kernel, desc_.dtype_, res.data_ptr(), output_bias_,
                               m, n, desc_.stream);
                }
            }
            #ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
            #endif
         }

        void MaskedMultiHeadAttention::kv_transpose(torch::Tensor& d_K_buf, torch::Tensor& d_V_buf,Buffer& K_buf,Buffer& V_buf)
         {
            RUN_KERNEL(copyKV_transpose_kernel,desc_.dtype_,d_K_buf.data_ptr(), d_V_buf.data_ptr(),K_buf.data_ptr(), V_buf.data_ptr(),cur_batch_size_, cur_seq_len_,
                   desc_.head_num_, size_per_head_);
            #ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
            #endif
         }
        
        void MaskedMultiHeadAttention::masked_attention(const Buffer& qkv_buffer,
                                                        Buffer& context_buf,
                                                        const int64_t *padding_len,
                                                        const int64_t *reorder_index,
                                                        const void* relative_attention_bias_)
         {
            RUN_KERNEL(fused_masked_attention_dispatch,desc_.dtype_,qkv_buffer.data_ptr(), q_bias_,k_cache_.data_ptr(),k_bias_,v_cache_.data_ptr(),v_bias_,
                        context_buf.data_ptr(),cur_batch_size_,first_batch_size_,desc_.head_num_,size_per_head_,step_, desc_.stream, padding_len,reorder_index, relative_attention_bias_);
            #ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
            #endif
         }
    
    }
}
