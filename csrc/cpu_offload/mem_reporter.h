//
// Created by ligz on 2021/4/28.
//

#ifndef EET_MEN_REPORTER_H
#define EET_MEN_REPORTER_H

#include "weight_attr.h"
#include "common.h"
//co : cpu_offset
namespace eet::co {

//get parameter size according to meta_desc
    class ParamCounter {
    public:
        ParamCounter(const MetaDesc& meta):meta_(meta) {
        }
        virtual size_t get_param_size(bool norm_before = false) = 0;
        virtual size_t get_buffer_size(bool norm_before = false) = 0;
        virtual size_t get_cache_size() = 0;

    protected:
        MetaDesc meta_;
    };

    class MultiHeadAttnPC : public ParamCounter{
    public:
        MultiHeadAttnPC(const MetaDesc& meta):ParamCounter(meta),q_weight_attr_(meta),
                                                q_bias_attr_(meta),k_weight_attr_(meta),
                                                k_bias_attr_(meta),v_weight_attr_(meta),
                                                v_bias_attr_(meta),out_weight_attr_(meta),
                                                out_bias_attr_(meta),ln_weight_attr_(meta),
                                                ln_bias_attr_(meta){
        }
        size_t get_param_size(bool norm_before=true){
            int itemsize = get_itemsize(meta_);
            size_t qkv_weights       = q_weight_attr_.size() + k_weight_attr_.size() + v_weight_attr_.size();
            size_t qkv_bias          = q_bias_attr_.size() + k_bias_attr_.size() + v_bias_attr_.size();
            size_t project_weights   = out_weight_attr_.size();
            size_t project_bias      = out_bias_attr_.size();
            size_t ln_beta           = ln_weight_attr_.size();
            size_t ln_gamma          = ln_bias_attr_.size();
            if(!norm_before){
                ln_beta  = 0;
                ln_gamma = 0;
            }
            size_t total = qkv_weights + qkv_bias + project_weights
                           + project_bias + ln_beta + ln_gamma;
            return total * itemsize * meta_.layer_num_;
        }

        size_t get_buffer_size(bool norm_before=true){
            int hidden_unit = meta_.hidden_units_;
            int itemsize = get_itemsize(meta_);
            size_t qkv_buffer       = meta_.batch_size_ * meta_.max_full_seq_len_ * hidden_unit * 3;
            size_t qkv_buf         = meta_.batch_size_ * meta_.max_full_seq_len_ * hidden_unit * 3;
            size_t qk_buf          = meta_.batch_size_ * meta_.head_num_ *
                                     meta_.max_full_seq_len_ * meta_.max_full_seq_len_;
            size_t left_padding_len= meta_.batch_size_;
            size_t total = qkv_buffer + qkv_buf + qk_buf + left_padding_len;
            return total * itemsize;
        }

        size_t get_cache_size(){
            int hidden_unit = meta_.hidden_units_;
            int itemsize = get_itemsize(meta_);
            // for some reason like add_residual, two buffers are requested
            size_t output   = meta_.batch_size_ * hidden_unit * meta_.max_full_seq_len_  * 2;
            size_t kv_cache = meta_.batch_size_ * meta_.max_seq_len_ * hidden_unit * 2;
            return kv_cache * itemsize * meta_.layer_num_ + output * itemsize;
        }

    private:
        AttnQWeightAttr q_weight_attr_;
        AttnQBiasAttr   q_bias_attr_;
        AttnKWeightAttr k_weight_attr_;
        AttnKBiasAttr   k_bias_attr_;
        AttnVWeightAttr v_weight_attr_;
        AttnVBiasAttr   v_bias_attr_;
        AttnOutWeightAttr out_weight_attr_;
        AttnOutBiasAttr   out_bias_attr_;
        AttnLnWeightAttr  ln_weight_attr_;
        AttnLnBiasAttr    ln_bias_attr_;
    };

    class EmbeddingPC : public ParamCounter{
    public:
        EmbeddingPC(const MetaDesc& meta, const int& vocab_size):ParamCounter(meta),
                        embeddingWeightAttr_(meta, vocab_size){
        }

        size_t get_param_size(bool norm){
            int itemsize = get_itemsize(meta_);
            size_t word_embedding = embeddingWeightAttr_.size();
            return word_embedding * itemsize;
        }
        size_t get_buffer_size(bool norm_before){
            return 0;
        }
        size_t get_cache_size(){
            int hidden_unit = meta_.hidden_units_;
            int itemsize = get_itemsize(meta_);
            size_t output   = meta_.batch_size_ * hidden_unit;
            return output * itemsize;
        }
    private:
        EmbeddingWeightAttr embeddingWeightAttr_;
    };

    class FfnPC : public ParamCounter{
    public:
        FfnPC(const MetaDesc& meta):ParamCounter(meta), inter_weight_attr_(meta),
                                inter_bias_attr_(meta),out_weight_attr_(meta),
                                out_bias_attr_(meta),ln_weight_attr_(meta),
                                ln_bias_attr_(meta){
            std::cout << "FFN Parameter Counter" << std::endl;
        }
        size_t get_param_size(bool norm=true){
            int hidden_unit = meta_.hidden_units_;
            int itemsize = get_itemsize(meta_);
            size_t ffn1_weights  = inter_weight_attr_.size();
            size_t ffn1_bias     = inter_bias_attr_.size();
            size_t ffn2_weights  = out_weight_attr_.size();
            size_t ffn2_bias     = out_bias_attr_.size();
            size_t ln_beta       = ln_weight_attr_.size();
            size_t ln_gamma      = ln_bias_attr_.size();
            if(!norm){
                ln_beta  = 0;
                ln_gamma = 0;
            }
            size_t total = ffn1_weights + ffn1_bias +
                           ffn2_weights + ffn2_bias + ln_gamma + ln_beta;
            return total * itemsize * meta_.layer_num_;
        }
        //for we reuse the ffn buffer with the qk_buffer, so it's 0
        size_t get_buffer_size(bool norm_before=true){
//            int itemsize = get_itemsize(meta_);
//            int hidden_unit = meta_.hidden_units_;
//            size_t ffn_inner =  meta_.batch_size_ * meta_.max_full_seq_len_ * hidden_unit * 4;
//            return ffn_inner * itemsize;
              return 0;
        }
        size_t get_cache_size(){
            return 0;
        }

    private:
        FFNInterWeightAttr inter_weight_attr_;
        FFNInterBiasAttr   inter_bias_attr_;
        FFNOutWeightAttr   out_weight_attr_;
        FFNOutBiasAttr     out_bias_attr_;
        FFNLnWeightAttr    ln_weight_attr_;
        FFNLnBiasAttr      ln_bias_attr_;
    };



    class MemReporter{
    public:
        MemReporter(const MetaDesc& meta, const int& vocab_size):meta_(meta),
                                                            embeddingPc_(meta, vocab_size),
                                                            multiHeadAttnPc_(meta),
                                                            ffnPc_(meta){
        }

        size_t get_total_buffer(){
           return embeddingPc_.get_buffer_size(false) +
                    multiHeadAttnPc_.get_buffer_size(true) +
                    ffnPc_.get_buffer_size(true);
        }

        size_t get_total_cache(){
            return embeddingPc_.get_cache_size() +
                    multiHeadAttnPc_.get_cache_size() +
                    ffnPc_.get_cache_size();
        }

        void print(){
            size_t emb_param = embeddingPc_.get_param_size(false);
            size_t emb_buff  = embeddingPc_.get_buffer_size(false);
            size_t emb_cache = embeddingPc_.get_cache_size();
            size_t attn_param = multiHeadAttnPc_.get_param_size(true);
            size_t attn_buff  = multiHeadAttnPc_.get_buffer_size(true);
            size_t attn_cache = multiHeadAttnPc_.get_cache_size();
            size_t ffn_param  = ffnPc_.get_param_size(true);
            size_t ffn_buff   = ffnPc_.get_buffer_size(true);
            size_t ffn_cache  = ffnPc_.get_cache_size();
            size_t itemsize = get_itemsize(meta_);
            std::cout << "itemsize : " << itemsize << std::endl;
            float total = (emb_param + emb_buff + emb_cache +
                           attn_param + attn_buff + attn_cache +
                           ffn_param + ffn_buff + ffn_cache) * 0.01; //for precent, * 0.01
            std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3);
            std::cout << "total memory : " << (total * 100) / ONE_G<< " G" << std::endl;
            std::cout << "total parameter : " << (emb_param + attn_param + ffn_param) / itemsize << std::endl;
            std::cout << "print GPU memory usage detail : " << std::endl;
            std::cout << "Embedding Layer * 1 " << std::endl;
            std::cout << "param : " << std::setw(13) << emb_param / itemsize << " (" << std::setw(6) << emb_param / ONE_G << "G, "
                      << std::setw(8) << emb_param / total << "% )   ";
            std::cout << "buff  : " << std::setw(13) <<  emb_buff / itemsize << " (" << std::setw(6) << emb_buff / ONE_G << "G, "
                      << std::setw(8) << emb_buff / total << "% )   ";
            std::cout << "cache: " << std::setw(13) << emb_cache / itemsize << " (" << std::setw(6) << emb_cache / ONE_G << "G, "
                      << std::setw(8) << emb_cache / total << "% )   " << std::endl;
            std::cout << "Attention Layer *  " << meta_.layer_num_ << std::endl;
            std::cout << "param : " << std::setw(13) << attn_param / itemsize << " (" << std::setw(6) << attn_param / ONE_G << "G, "
                      << std::setw(8) << attn_param / total << "% )   ";
            std::cout << "buff  : " << std::setw(13) << attn_buff / itemsize << " (" << std::setw(6) << attn_buff / ONE_G << "G, "
                      << std::setw(8) << attn_buff / total << "% )   ";
            std::cout << "cache: " << std::setw(13) << attn_cache / itemsize << " (" << std::setw(6) << attn_cache / ONE_G << "G, "
                      << std::setw(8) << attn_cache / total << "% )   " << std::endl;
            std::cout << "FFN Layer *  " << meta_.layer_num_ << std::endl;
            std::cout << "param : " << std::setw(13) << ffn_param / itemsize << " (" << std::setw(6) << ffn_param / ONE_G << "G, "
                      << std::setw(8) << ffn_param / total << "% )   ";
            std::cout << "buff  : " << std::setw(13) << ffn_buff / itemsize << " (" << std::setw(6) << ffn_buff / ONE_G << "G, "
                      << std::setw(8) << ffn_buff / total << "% )   ";
            std::cout << "cache: " << std::setw(13) << ffn_cache / itemsize << " (" << std::setw(6) << ffn_cache / ONE_G << "G, "
                      << std::setw(8) << ffn_cache / total << "% )   " << std::endl;
            return;
        }
    private:
        MetaDesc meta_;
        EmbeddingPC embeddingPc_;
        MultiHeadAttnPC multiHeadAttnPc_;
        FfnPC ffnPc_;
    };
}

#endif //EET_MEN_REPORTER_H
