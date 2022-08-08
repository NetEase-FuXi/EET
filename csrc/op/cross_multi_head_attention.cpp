#include "op/cross_multi_head_attention.hpp"
#include "core/add_bias.cuh"
#include "core/transpose.cuh"
#include "core/layer_norm.cuh"
#include "core/cross_add_bias.cuh"
#include "core/attention_dispatch.cuh"
#include "core/gpt2_cross_softmax.cuh"
#include <iostream>
namespace eet{
    namespace op{
        CrossMultiHeadAttention::CrossMultiHeadAttention(MetaDesc desc,
                                    const torch::Tensor& Q_weights,
                                    const torch::Tensor& K_weights,
                                    const torch::Tensor& V_weights,
                                    const torch::Tensor& Q_bias,
                                    const torch::Tensor& K_bias,
                                    const torch::Tensor& V_bias,
                                    const torch::Tensor& Output_weights,
                                    const torch::Tensor& Output_bias,
                                    const torch::Tensor& layernorm_weights,
                                    const torch::Tensor& layernorm_bias):
            desc_(desc),
            step_(1),
            q_weights_(Q_weights.data_ptr()),
            k_weights_(K_weights.data_ptr()),
            v_weights_(V_weights.data_ptr()),
            q_bias_(Q_bias.data_ptr()),
            k_bias_(K_bias.data_ptr()),
            v_bias_(V_bias.data_ptr()),
            output_weights_(Output_weights.data_ptr()),
            output_bias_(Output_bias.data_ptr()),
            layernorm_weights_(layernorm_weights.data_ptr()),
            layernorm_bias_(layernorm_bias.data_ptr())
        {
            with_bias_ = q_bias_ != nullptr ? true : false;
            // size_per_head_ = 64;
            size_per_head_ = desc_.hidden_units_ / desc_.head_num_;
            inner_dim_ = size_per_head_* desc_.head_num_;
            // output_ = torch::zeros({desc_.batch_size_ , desc_.max_full_seq_len_ ,desc_.hidden_units_}, desc_.options_);
            key_mem_cache_ =torch::zeros({desc_.batch_size_ , desc_.max_seq_len_ ,desc_.hidden_units_}, desc_.options_);
            value_mem_cache_ = torch::zeros_like(key_mem_cache_);
            Buffer& emb_ffn_out = MManager::get_instance().get_cache(desc_.batch_size_ * desc_.max_full_seq_len_ * desc_.hidden_units_, desc_.dtype_, desc_.options_,"cross_attn");

            switch(desc_.dtype_){
                case torch::kFloat32:
                    qkv_weights_algo_ = CUBLAS_GEMM_DEFAULT;
                    q_k_algo_ = CUBLAS_GEMM_DEFAULT;
                    attn_v_algo_ = CUBLAS_GEMM_DEFAULT;
                    alpha_ = new float();
                    beta_  = new float();
                    *((float*)alpha_) = 1.0f;
                    *((float*)beta_)  = 0.0f;
                    break;
                case torch::kFloat16:
                    qkv_weights_algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
                    q_k_algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
                    attn_v_algo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
                    alpha_ = new half();
                    beta_  = new half();
                    *((half*)alpha_) = (half)1.0f;
                    *((half*)beta_)  = (half)0.0f;
                    break;
                //TODO
                case torch::kInt8:
                    break;
            }
        }

        torch::Tensor CrossMultiHeadAttention::forward(torch::Tensor& input,
                                    torch::Tensor& memory,
                                    const torch::Tensor& pre_padding_length,
                                    bool pre_layernorm,
                                    bool add_residual,
                                    const torch::Tensor& length_per_sample,
                                    bool first_pass){
            if(first_pass)
            {
                return forward_full(input,memory,pre_padding_length,pre_layernorm,add_residual);
            }
            else
            {
                return forward_inc(input,memory,pre_layernorm,add_residual,length_per_sample);
            }
        }

        // full decoder
        torch::Tensor CrossMultiHeadAttention::forward_full(torch::Tensor& input,
                            torch::Tensor& memory,
                            const torch::Tensor& pre_padding_length,
                            bool pre_layernorm,
                            bool add_residual){
            assert((input.dtype() == desc_.dtype_) && "input's dtype is not the same as CrossMultiHeadAttention's dtype");
            cur_batch_size_ = input.sizes()[0];
            cur_seq_len_ = input.sizes()[1];
            mem_seq_len_ = memory.sizes()[1];
            step_ = 1;
            //qkv * weights
            Buffer& q_buffer = MManager::get_instance().get_buffer(desc_.batch_size_ *  desc_.max_full_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_);
            Buffer& k_buffer = MManager::get_instance().get_buffer(desc_.batch_size_ *  desc_.max_full_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_);
            Buffer& v_buffer = MManager::get_instance().get_buffer(desc_.batch_size_ *  desc_.max_full_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_);
            
            if(pre_layernorm)
            {
                // pre_layerNorm
                Buffer& layernormed_query = MManager::get_instance().get_buffer(desc_.batch_size_ *  desc_.max_full_seq_len_ *
                        desc_.hidden_units_, desc_.dtype_, desc_.options_);
                layer_norm(input,layernormed_query);

                //qkv * weights
                qkv_weights_mul(layernormed_query.data_ptr(), memory,q_buffer,k_buffer,v_buffer);
                layernormed_query.free();
            }
            else{
                //qkv * weights
                qkv_weights_mul(input.data_ptr(),memory, q_buffer,k_buffer,v_buffer);
            }

            //qkv add bias                
            Buffer& q_buf = MManager::get_instance().get_buffer(desc_.batch_size_ *  desc_.max_full_seq_len_ *
                                    desc_.hidden_units_, desc_.dtype_, desc_.options_);
            Buffer& k_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ *
                                    desc_.hidden_units_, desc_.dtype_, desc_.options_);
            Buffer& v_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.max_seq_len_ *
                                    desc_.hidden_units_, desc_.dtype_, desc_.options_);
            qkv_add_bias(q_buffer, k_buffer, v_buffer, q_buf, k_buf, v_buf);

            q_buffer.free();
            k_buffer.free();
            v_buffer.free();

            //q * k
            Buffer& qk_buf = MManager::get_instance().get_buffer(desc_.batch_size_ * desc_.head_num_ *
                                         desc_.max_full_seq_len_ * desc_.max_seq_len_, desc_.dtype_, desc_.options_);
            q_k_mul(q_buf, k_buf, qk_buf);
            q_buf.free();

            //softmax
            qk_softmax(qk_buf,pre_padding_length);

            //attn * v
            Buffer& transpose_dst = MManager::get_instance().get_buffer(desc_.batch_size_ *  desc_.max_full_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_);
            
            attn_v_mul(qk_buf,v_buf,transpose_dst);

            qk_buf.free();

            // transpose k\v cache
            kv_transpose(key_mem_cache_,value_mem_cache_,k_buf,v_buf);


            k_buf.free();
            v_buf.free();

            //transpose
            Buffer& dst = MManager::get_instance().get_buffer(desc_.batch_size_ *  desc_.max_full_seq_len_ *
                                    inner_dim_, desc_.dtype_, desc_.options_);

            transpose(transpose_dst, dst);
            transpose_dst.free();

            Buffer& output = MManager::get_instance().get_cache(desc_.batch_size_ * desc_.max_full_seq_len_ * desc_.hidden_units_, desc_.dtype_, desc_.options_,"cross_attn");

            //project
            project(dst,output,input ,pre_layernorm,add_residual);
            // project(dst, output_);
            dst.free();
            step_ = cur_seq_len_;

            auto res = torch::from_blob(output.data_ptr(), input.sizes(), input.strides(), desc_.options_);
            return std::move(res);
        }
        
        // incremental decoder
        torch::Tensor CrossMultiHeadAttention::forward_inc(torch::Tensor &input,
                                                           torch::Tensor &memory,
                                                           bool pre_layernorm,
                                                           bool add_residual,
                                                           const torch::Tensor &length_per_sample) {
            assert((input.dtype() == desc_.dtype_) && "input's dtype is not the same as CrossMultiHeadAttention's dtype");
            step_ += 1;
            cur_batch_size_ = input.sizes()[0];
            cur_seq_len_ = input.sizes()[1];
            //qkv * weights
            Buffer& q_buffer = MManager::get_instance().get_buffer(desc_.batch_size_ *  desc_.max_full_seq_len_ *
                                    desc_.hidden_units_, desc_.dtype_, desc_.options_);

            // not use
            Buffer& k_buffer = MManager::get_instance().get_buffer(desc_.batch_size_ *  desc_.max_full_seq_len_ *
                                    desc_.hidden_units_, desc_.dtype_, desc_.options_);
            Buffer& v_buffer = MManager::get_instance().get_buffer(desc_.batch_size_ *  desc_.max_full_seq_len_ *
                                    desc_.hidden_units_, desc_.dtype_, desc_.options_);

            
            if(pre_layernorm)
            {
                // pre_layerNorm
                Buffer& layernormed_query = MManager::get_instance().get_buffer(desc_.batch_size_ *  desc_.max_full_seq_len_ *
                        desc_.hidden_units_, desc_.dtype_, desc_.options_);
                layer_norm(input,layernormed_query);

                //qkv * weights
                qkv_weights_mul(layernormed_query.data_ptr(), memory,q_buffer,k_buffer,v_buffer);
                layernormed_query.free();
            }
            else{
                //qkv * weights
                qkv_weights_mul(input.data_ptr(),memory, q_buffer,k_buffer,v_buffer);
            }

            Buffer& context_buf = MManager::get_instance().get_buffer(desc_.batch_size_ *  desc_.max_full_seq_len_ *
                                    desc_.hidden_units_, desc_.dtype_, desc_.options_);


            //attention_dispatch
            attention_dispatch(q_buffer,length_per_sample,context_buf);
        
            q_buffer.free();
            k_buffer.free();
            v_buffer.free();
            Buffer& output = MManager::get_instance().get_cache(desc_.batch_size_ * desc_.max_full_seq_len_ * desc_.hidden_units_, desc_.dtype_, desc_.options_,"cross_attn");

            project(context_buf, output, input,pre_layernorm,add_residual);

            // project(context_buf, output_);
            context_buf.free();
            auto res = torch::from_blob(output.data_ptr(), input.sizes(), input.strides(), desc_.options_);
            return std::move(res);
        }

        // layerNorm
        void CrossMultiHeadAttention::layer_norm(const torch::Tensor& input_tensor, Buffer& layernorm_query)
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


        void CrossMultiHeadAttention::qkv_weights_mul(void* input, 
                                    torch::Tensor& memory, 
                                    Buffer& q_buffer,
                                    Buffer& k_buffer,
                                    Buffer& v_buffer){
               
                int m = cur_batch_size_ * cur_seq_len_;
                const int k = desc_.hidden_units_;
                const int n = inner_dim_;
                check_cuda_error(cublasGemmEx(desc_.cublasHandle,
                        CUBLAS_OP_N, CUBLAS_OP_N, 
                        n, m, k, 
                        alpha_, 
                        q_weights_, desc_.computeType_, n, 
                        input, desc_.computeType_, k, 
                        beta_, 
                        q_buffer.data_ptr(), desc_.computeType_, n, 
                        desc_.computeType_, 
                        qkv_weights_algo_));

    #ifdef _DEBUG_MODE_
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
    #endif      
                if(step_ == 1)
                {   
                    m = cur_batch_size_ * mem_seq_len_;
                    check_cuda_error(cublasGemmEx(desc_.cublasHandle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        n, m, k, 
                        alpha_, 
                        k_weights_, desc_.computeType_, n, 
                        memory.data_ptr(), desc_.computeType_, k, 
                        beta_, 
                        k_buffer.data_ptr(), desc_.computeType_, n, 
                        desc_.computeType_, 
                        qkv_weights_algo_));

    #ifdef _DEBUG_MODE_
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
    #endif
                check_cuda_error(cublasGemmEx(desc_.cublasHandle,
                        CUBLAS_OP_N, CUBLAS_OP_N, 
                        n, m, k,
                        alpha_,
                        v_weights_, desc_.computeType_, n, 
                        memory.data_ptr(), desc_.computeType_, k, 
                        beta_, 
                        v_buffer.data_ptr(), desc_.computeType_, n, 
                        desc_.computeType_, 
                        qkv_weights_algo_));
                
    #ifdef _DEBUG_MODE_
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
    #endif
                }
        }

        void CrossMultiHeadAttention::qkv_add_bias(const Buffer& q_buffer,
                                    const Buffer& k_buffer,
                                    const Buffer& v_buffer,
                                    Buffer& q_buf,
                                    Buffer& k_buf,
                                    Buffer& v_buf){
            RUN_KERNEL(add_QKV_bias_cross_opt_kernel,desc_.dtype_,q_buffer.data_ptr(), q_bias_,
                        k_buffer.data_ptr(), k_bias_, v_buffer.data_ptr(), v_bias_,
                        q_buf.data_ptr(), k_buf.data_ptr(), v_buf.data_ptr(),
                        cur_batch_size_, cur_seq_len_, mem_seq_len_,desc_.head_num_, size_per_head_, desc_.stream);
#ifdef _DEBUG_MODE_
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif
        }

        void CrossMultiHeadAttention::q_k_mul(const Buffer& q_buf, const Buffer& k_buf,
                                                Buffer& qk_buf){
            check_cuda_error(cublasGemmStridedBatchedEx(MetaDesc::cublasHandle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                mem_seq_len_, cur_seq_len_, size_per_head_,
                alpha_,
                k_buf.data_ptr(), desc_.computeType_, size_per_head_, mem_seq_len_ * size_per_head_,
                q_buf.data_ptr(), desc_.computeType_, size_per_head_, cur_seq_len_ * size_per_head_,
                beta_,
                qk_buf.data_ptr(), desc_.computeType_, mem_seq_len_, mem_seq_len_ * cur_seq_len_,
                cur_batch_size_ * desc_.head_num_,
                desc_.computeType_,
                q_k_algo_));

#ifdef _DEBUG_MODE_
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif
        }

        void CrossMultiHeadAttention::qk_softmax(Buffer& qk_buf, const torch::Tensor& padding_index){
            float scalar = 1.0f;
            if (with_bias_)
                scalar = 1 / sqrtf(size_per_head_ * 1.0f);
            RUN_KERNEL(cross_softmax_kernel,desc_.dtype_,qk_buf.data_ptr(), cur_batch_size_,
                    desc_.head_num_,cur_seq_len_, mem_seq_len_, scalar, desc_.stream);
#ifdef _DEBUG_MODE_
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif 
        }

        void CrossMultiHeadAttention::attn_v_mul(const Buffer& qk_buf,
                                             const Buffer& v_buf,
                                             Buffer& transpose_dst){
            check_cuda_error(cublasGemmStridedBatchedEx(desc_.cublasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    size_per_head_, cur_seq_len_, mem_seq_len_,
                    alpha_,
                    v_buf.data_ptr(), desc_.computeType_, size_per_head_, mem_seq_len_ * size_per_head_,
                    qk_buf.data_ptr(), desc_.computeType_, mem_seq_len_, mem_seq_len_ * cur_seq_len_,
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

        void CrossMultiHeadAttention::transpose(const Buffer& transpose_dst, Buffer&  dst){
            RUN_KERNEL(transpose_kernel,desc_.dtype_,transpose_dst.data_ptr(),dst.data_ptr(), cur_batch_size_, cur_seq_len_,
                    desc_.head_num_, size_per_head_, desc_.stream);

#ifdef _DEBUG_MODE_
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif
        }

        void CrossMultiHeadAttention::project(const Buffer& dst,Buffer& res,torch::Tensor& input, bool pre_layernorm,bool add_residual){
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

        void CrossMultiHeadAttention::kv_transpose(torch::Tensor& d_K_buf, torch::Tensor& d_V_buf,Buffer& K_buf,Buffer& V_buf)
         {
            RUN_KERNEL(copyKV_transpose_cross_kernel,desc_.dtype_,d_K_buf.data_ptr(), d_V_buf.data_ptr(),K_buf.data_ptr(), V_buf.data_ptr(),cur_batch_size_, mem_seq_len_,
                   desc_.head_num_, size_per_head_);
            #ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
            #endif
         }
        
        void CrossMultiHeadAttention::attention_dispatch(const Buffer& q_buffer,
                                                        const torch::Tensor& length_per_sample,
                                                        Buffer& context_buf
                                                        )
         {
            RUN_KERNEL(cross_attention_dispatch,desc_.dtype_,q_buffer.data_ptr(),
                        q_bias_,key_mem_cache_.data_ptr(),k_bias_,value_mem_cache_.data_ptr(),v_bias_,(int *)length_per_sample.data_ptr(),
                        context_buf.data_ptr(),cur_batch_size_,desc_.head_num_,size_per_head_,step_,mem_seq_len_, desc_.stream);
                        
            #ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
            #endif
         }
    } // namespace op
} // namespace eet