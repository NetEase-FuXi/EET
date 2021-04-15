#include "op/embedding.hpp"
// #include "core/common.cuh"
#include "core/layer_norm.cuh"
#include "core/embedding_kernel.cuh"

namespace eet
{
    namespace op
    {
        Embedding::Embedding(MetaDesc desc, 
                            const torch::Tensor &embedding_weights, 
                            const torch::Tensor &position_weights, 
                            const torch::Tensor &token_type_weights,
                            const torch::Tensor& layernorm_weights, 
                            const torch::Tensor& layernorm_bias) : 
        step_(0),
        desc_(desc),
        embedding_weights_(embedding_weights.data_ptr()),
        position_weights_(position_weights.data_ptr()),
        token_type_weights_(token_type_weights.data_ptr()),
        layernorm_weights_(layernorm_weights.data_ptr()),
        layernorm_bias_(layernorm_bias.data_ptr()),
        cur_batch_size_(0),
        cur_seq_len_(0)
        {
            output_ = torch::zeros({desc_.batch_size_, desc_.max_full_seq_len_, desc_.hidden_units_}, desc_.options_);
        }

        // embed tokens and positions
        torch::Tensor Embedding::forward_fairseq(const torch::Tensor &input_tensor,bool no_scale_embedding,int padding_idx)
        {
            cur_batch_size_ = input_tensor.sizes()[0];
            cur_seq_len_ = input_tensor.sizes()[1];
            int embedding_num = cur_batch_size_ * cur_seq_len_;

            // printf("cur_batch_size_:%d cur_seq_len_:%d embedding_num:%d padding_idx:%d desc_.hidden_units_:%d  no_scale_embedding:%d\n",
            //             cur_batch_size_,cur_seq_len_,embedding_num,padding_idx,desc_.hidden_units_,no_scale_embedding);
            // step_ = cur_batch_size_;
            const int64_t *input_ids = input_tensor.data_ptr<int64_t>();

            RUN_KERNEL(embedding_lookup_kernel, desc_.dtype_,embedding_weights_, input_ids,output_.data_ptr(),
                        embedding_num, desc_.hidden_units_, desc_.stream,/*acc=*/false,/*ratio=*/1,no_scale_embedding);
            
            #ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
            #endif
            int m = embedding_num;
            int n = desc_.hidden_units_;
            RUN_KERNEL(position_encoding_kernel, desc_.dtype_,output_.data_ptr(),
                        cur_seq_len_,step_, padding_idx,m , n, desc_.stream);

            #ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
            #endif
            step_ += cur_seq_len_;

            auto res = torch::from_blob(output_.data_ptr(),{cur_batch_size_, cur_seq_len_,desc_.hidden_units_} ,desc_.options_);
            return std::move(res);
        }


        // embedding_lookup + positional_encoding + token_type
        torch::Tensor Embedding::forward_transformers(const torch::Tensor &input_tensor,const torch::Tensor &position_ids,const torch::Tensor &token_type_ids, bool if_layernorm)
        {
            cur_batch_size_ = input_tensor.sizes()[0];
            cur_seq_len_ = input_tensor.sizes()[1];

            if(if_layernorm)
            {
                // embedding
                Buffer& embedding_out = MManager::get_instance().get_buffer(cur_batch_size_ * cur_seq_len_ *
                                                        desc_.hidden_units_, desc_.dtype_, desc_.options_);
                fused_embedding(input_tensor,position_ids,token_type_ids,embedding_out.data_ptr());
                
                // layernorm
                layer_norm(embedding_out,output_);
                embedding_out.free();
                auto res = torch::from_blob(output_.data_ptr(),{cur_batch_size_, cur_seq_len_,desc_.hidden_units_} ,desc_.options_);
                return std::move(res);
            }
            else
            {
                // embedding                    
                fused_embedding(input_tensor,position_ids,token_type_ids,output_.data_ptr());
                auto res = torch::from_blob(output_.data_ptr(),{cur_batch_size_, cur_seq_len_,desc_.hidden_units_} ,desc_.options_);
                return std::move(res);
            }
        }

        void Embedding::fused_embedding(const torch::Tensor& input_tensor,const torch::Tensor &position_ids,const torch::Tensor &token_type_ids,void* embedding_out)
        {
            int tensor_col = input_tensor.sizes()[0];
            int tensor_row = input_tensor.sizes()[1];
            int embedding_num = tensor_col * tensor_row;

            tensor_col = position_ids.sizes()[0];
            tensor_row = position_ids.sizes()[1];
            int position_num = tensor_col * tensor_row;
            int position_ratio  = embedding_num / position_num;

            bool no_scale_embedding = true;
            // embedding
            const int64_t *input_ids = input_tensor.data_ptr<int64_t>();
            RUN_KERNEL(embedding_lookup_kernel, desc_.dtype_,embedding_weights_, input_ids,embedding_out,
                    embedding_num, desc_.hidden_units_, desc_.stream,/*acc=*/false,1,no_scale_embedding);

            
            const int64_t *token_type = token_type_ids.data_ptr<int64_t>();
            if(token_type_ids.data_ptr() != nullptr)
            {   
                tensor_col = token_type_ids.sizes()[0];
                tensor_row = token_type_ids.sizes()[1];
                int token_num = tensor_col * tensor_row;
                int token_type_ratio = embedding_num / token_num;
                RUN_KERNEL(embedding_lookup_kernel, desc_.dtype_,token_type_weights_, token_type,embedding_out,
                        token_num, desc_.hidden_units_, desc_.stream,/*acc=*/true,token_type_ratio,no_scale_embedding);
            }

            const int64_t *position = position_ids.data_ptr<int64_t>();
            RUN_KERNEL(embedding_lookup_kernel, desc_.dtype_,position_weights_, position,embedding_out,
                    position_num, desc_.hidden_units_, desc_.stream,/*acc=*/true,position_ratio,no_scale_embedding);


#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

        void Embedding::layer_norm(Buffer& input,torch::Tensor& out)
        {
            const int m = cur_batch_size_ * cur_seq_len_;
            int n = desc_.hidden_units_;
            RUN_KERNEL(layernorm,desc_.dtype_,input.data_ptr(),layernorm_weights_,layernorm_bias_,out.data_ptr(), m, n, desc_.stream);
#ifdef _DEBUG_MODE_
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
        }

    } // namespace op
} // namespace eet