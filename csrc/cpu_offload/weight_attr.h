//
// Created by ligz on 2021/4/28.
//

#ifndef EET_WEIGHT_ATTR_H
#define EET_WEIGHT_ATTR_H
#include "../op/op.h"

#include "common.h"
#include "mem_reporter.h"


namespace eet{
    namespace co{

        typedef enum WeightType{
            Unknown,
            WordEmbedding,
            AttnQWeight,
            AttnKWeight,
            AttnVWeight,
            AttnQBias,
            AttnKBias,
            AttnVBias,
            AttnOutWeight,
            AttnOutBias,
            AttnLnWeight,
            AttnLnBias,
            FfnInterWeight,
            FfnInterBias,
            FfnOutWeight,
            FfnOutBias,
            FfnLnWeight,
            FfnLnBias
        } WeightType;

        inline WeightType string2wtype(const std::string& weight_name){
            if(weight_name == "word_embedding")
                return WordEmbedding;
            if(weight_name == "attn_q_weights")
                return AttnQWeight;
            if(weight_name == "attn_k_weights")
                return AttnKWeight;
            if(weight_name == "attn_v_weights")
                return AttnVWeight;
            if(weight_name == "attn_q_bias")
                return AttnQBias;
            if(weight_name == "attn_k_bias")
                return AttnKBias;
            if(weight_name == "attn_v_bias")
                return AttnVBias;
            if(weight_name == "attn_output_weights")
                return AttnOutWeight;
            if(weight_name == "attn_output_bias")
                return AttnOutBias;
            if(weight_name == "attn_ln_weights")
                return AttnLnWeight;
            if(weight_name == "attn_ln_bias")
                return AttnLnBias;
            if(weight_name == "ffn_inter_weights")
                return FfnInterWeight;
            if(weight_name == "ffn_inter_bias")
                return FfnInterBias;
            if(weight_name == "ffn_out_weights")
                return FfnOutWeight;
            if(weight_name == "ffn_out_bias")
                return FfnOutBias;
            if(weight_name == "ffn_ln_weights")
                return FfnLnWeight;
            if(weight_name == "ffn_ln_bias")
                return FfnLnBias;
            return Unknown;
        };

        inline std::string wtype2string(const WeightType& type){
            switch(type){
                case WordEmbedding:
                    return "word_embedding";
                case AttnQWeight:
                    return "attn_q_weights";
                case AttnKWeight:
                    return "attn_k_weights";
                case AttnVWeight:
                    return "attn_v_weights";
                case AttnQBias:
                    return "attn_q_bias";
                case AttnKBias:
                    return "attn_k_bias";
                case AttnVBias:
                    return "attn_v_bias";
                case AttnOutWeight:
                    return "attn_output_weights";
                case AttnOutBias:
                    return "attn_output_bias";
                case AttnLnWeight:
                    return "attn_ln_weights";
                case AttnLnBias:
                    return "attn_ln_bias";
                case FfnInterWeight:
                    return "ffn_inter_weights";
                case FfnInterBias:
                    return "ffn_inter_bias";
                case FfnOutWeight:
                    return "ffn_out_weights";
                case FfnOutBias:
                    return "ffn_out_bias";
                case FfnLnWeight:
                    return "ffn_ln_weights";
                case FfnLnBias:
                    return "ffn_ln_bias";
                case Unknown:
                    return "unknown_weight_type";
            }
        }

        //weight attribute
        //since embedding is always on GPU, it does not need inherit from this
        //Now attn and ffn must inherit from this
        struct WeightAttr{
            WeightAttr(const MetaDesc& meta):meta_(meta){
                check_cuda_error(cudaStreamCreate(&stream_));
                fixed_ = false;
            }
            virtual size_t size() const{
                std::cout << "warning: use the method in the subclass" << std::endl;
                return 0;
            };

            WeightType type_;
            bool on_device_;
            bool fixed_; // if fixed on device memory, true for embedding
            MetaDesc meta_;
            cudaStream_t stream_;

            std::vector<torch::Tensor> device_tensors; //store weight for every layer
            std::vector<torch::Tensor> host_tensors; // store weight for every layer on pinned memory
            virtual ~WeightAttr(){
                check_cuda_error(cudaStreamDestroy(stream_));
                device_tensors.clear();
                host_tensors.clear();
            }
        };

        struct EmbeddingWeightAttr: public WeightAttr{
            EmbeddingWeightAttr(const MetaDesc& meta, const int& vocab_size):WeightAttr(meta){
                vocab_size_ = vocab_size;
                on_device_ = true;
                type_ = WordEmbedding;
                fixed_ = true;
            }

            size_t size() const{
                return vocab_size_ * meta_.hidden_units_;
            }
            int vocab_size_ = 0;
            virtual ~EmbeddingWeightAttr(){
            }
        };

//        struct EmbeddingLnWeightAttr : public WeightAttr{
//            EmbeddingLnWeightAttr(const MetaDesc& meta):WeightAttr(meta){
//                name_ = "embedding_ln_weight";
//                on_device_ = false;
//            }
//            size_t size() const{
//                return meta_.hidden_units_;
//            }
//            virtual ~EmbeddingLnWeightAttr(){
//            }
//        };
//
//        struct EmbeddingLnBiasAttr : public WeightAttr{
//            EmbeddingLnBiasAttr(const MetaDesc& meta):WeightAttr(meta){
//                name_ = "attn_ln_bias";
//                on_device_ = false;
//            }
//            size_t size() const{
//                return meta_.hidden_units_;
//            }
//            virtual ~EmbeddingLnBiasAttr(){
//            }
//        };

        struct AttnQWeightAttr : public WeightAttr{
            AttnQWeightAttr(const MetaDesc& meta):WeightAttr(meta){
                type_ = AttnQWeight;
                on_device_ = false;
            }
            size_t  size() const{
                return meta_.hidden_units_ * meta_.hidden_units_;
            }
            virtual ~AttnQWeightAttr(){
            }
        };

        struct AttnKWeightAttr : public WeightAttr{
            AttnKWeightAttr(const MetaDesc& meta):WeightAttr(meta){
                type_ = AttnKWeight;
                on_device_ = false;
            }
            size_t size() const{
                return meta_.hidden_units_ * meta_.hidden_units_;
            }
            virtual ~AttnKWeightAttr(){
            }
        };

        struct AttnVWeightAttr : public WeightAttr{
            AttnVWeightAttr(const MetaDesc& meta):WeightAttr(meta){
                type_ = AttnVWeight;
                on_device_ = false;
            }
            size_t size() const{
                return meta_.hidden_units_ * meta_.hidden_units_;
            }
            virtual ~AttnVWeightAttr(){
            }
        };

        struct AttnQBiasAttr : public WeightAttr{
            AttnQBiasAttr(const MetaDesc& meta):WeightAttr(meta){
                type_ = AttnQBias;
                on_device_ = false;
            }
            size_t size() const{
                return meta_.hidden_units_;
            }
            virtual ~AttnQBiasAttr(){
            }
        };

        struct AttnKBiasAttr : public WeightAttr{
            AttnKBiasAttr(const MetaDesc& meta):WeightAttr(meta){
                type_ = AttnKBias;
                on_device_ = false;
            }
            size_t size() const{
                return meta_.hidden_units_;
            }
            virtual ~AttnKBiasAttr(){
            }
        };

        struct AttnVBiasAttr : public WeightAttr{
            AttnVBiasAttr(const MetaDesc& meta):WeightAttr(meta){
                type_ = AttnVBias;
                on_device_ = false;
            }
            size_t size() const{
                return meta_.hidden_units_;
            }
            virtual ~AttnVBiasAttr(){
            }
        };

        struct AttnOutWeightAttr : public WeightAttr{
            AttnOutWeightAttr(const MetaDesc& meta):WeightAttr(meta){
                type_ = AttnOutWeight;
                on_device_ = false;
            }
            size_t size() const{
                return meta_.hidden_units_ * meta_.hidden_units_;
            }
            virtual ~AttnOutWeightAttr(){
            }
        };

        struct AttnOutBiasAttr : public WeightAttr{
            AttnOutBiasAttr(const MetaDesc& meta):WeightAttr(meta){
                type_ = AttnOutBias;
                on_device_ = false;
            }
            size_t size() const{
                return meta_.hidden_units_;
            }
            virtual ~AttnOutBiasAttr(){
            }
        };

        struct AttnLnWeightAttr : public WeightAttr{
            AttnLnWeightAttr(const MetaDesc& meta):WeightAttr(meta){
                type_ = AttnLnWeight;
                on_device_ = false;
            }
            size_t size() const{
                return meta_.hidden_units_;
            }
            virtual ~AttnLnWeightAttr(){
            }
        };

        struct AttnLnBiasAttr : public WeightAttr{
            AttnLnBiasAttr(const MetaDesc& meta):WeightAttr(meta){
                type_ = AttnLnBias;
                on_device_ = false;
            }
            size_t size() const{
                return meta_.hidden_units_;
            }
            virtual ~AttnLnBiasAttr(){
            }
        };

        struct FFNInterWeightAttr : public WeightAttr{
            FFNInterWeightAttr(const MetaDesc& meta):WeightAttr(meta){
                type_ = FfnInterWeight;
                on_device_ = false;
            }
            size_t size() const{
                return meta_.hidden_units_ * meta_.hidden_units_ * 4;
            }
            virtual ~FFNInterWeightAttr(){
            }
        };

        struct FFNInterBiasAttr : public WeightAttr{
            FFNInterBiasAttr(const MetaDesc& meta):WeightAttr(meta){
                type_ = FfnInterBias;
                on_device_ = false;
            }
            size_t size() const{
                return meta_.hidden_units_ * 4;
            }
            virtual ~FFNInterBiasAttr(){
            }
        };

        struct FFNOutWeightAttr : public WeightAttr{
            FFNOutWeightAttr(const MetaDesc& meta):WeightAttr(meta){
                type_ = FfnOutWeight;
                on_device_ = false;
            }
            size_t size() const{
                return meta_.hidden_units_ * meta_.hidden_units_ * 4;
            }
            virtual ~FFNOutWeightAttr(){
            }
        };

        struct FFNOutBiasAttr : public WeightAttr{
            FFNOutBiasAttr(const MetaDesc& meta):WeightAttr(meta){
                    type_ = FfnOutBias;
                    on_device_ = false;
            }
            size_t size() const{
                return meta_.hidden_units_;
            }
            virtual ~FFNOutBiasAttr(){
            }
        };

        struct FFNLnWeightAttr : public WeightAttr{
            FFNLnWeightAttr(const MetaDesc& meta):WeightAttr(meta){
                type_ = FfnLnWeight;
                on_device_ = false;
            }
            size_t size() const{
                return meta_.hidden_units_;
            }
            virtual ~FFNLnWeightAttr(){
            }
        };

        struct FFNLnBiasAttr : public WeightAttr{
            FFNLnBiasAttr(const MetaDesc& meta):WeightAttr(meta){
                type_ = FfnLnBias;
                on_device_ = false;
            }
            size_t size() const{
                return meta_.hidden_units_;
            }
            virtual ~FFNLnBiasAttr(){
            }
        };
    }
}

#endif //EET_WEIGHT_ATTR_H
