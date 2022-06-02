//
// Created by ligz on 2021/4/30.
//

#ifndef EET_MEM_STRATEGY_H
#define EET_MEM_STRATEGY_H

#include "common.h"
#include "mem_reporter.h"

namespace eet::co{
    //Try to allocate the weights tensors to device memory one by one
    //prioritize large tensor
    class AllocationStrategy{
    public:
        AllocationStrategy(const MetaDesc& meta, const int& vocab_size):meta_(meta),
                                                                        thres_(0.95),
                                                                        mr_(meta, vocab_size){
            register_weight(new EmbeddingWeightAttr(meta, vocab_size));
            register_weight(new AttnQWeightAttr(meta));
            register_weight(new AttnKWeightAttr(meta));
            register_weight(new AttnVWeightAttr(meta));
            register_weight(new AttnQBiasAttr(meta));
            register_weight(new AttnKBiasAttr(meta));
            register_weight(new AttnVBiasAttr(meta));
            register_weight(new AttnOutWeightAttr(meta));
            register_weight(new AttnOutBiasAttr(meta));
            register_weight(new AttnLnWeightAttr(meta));
            register_weight(new AttnLnBiasAttr(meta));
            register_weight(new FFNInterWeightAttr(meta));
            register_weight(new FFNInterBiasAttr(meta));
            register_weight(new FFNOutWeightAttr(meta));
            register_weight(new FFNOutBiasAttr(meta));
            register_weight(new FFNLnWeightAttr(meta));
            register_weight(new FFNLnBiasAttr(meta));

            std::sort(attrs_.begin(), attrs_.end(),
                      [](const WeightAttr* wa1, const WeightAttr* wa2){
                          return (wa1->size() < wa2->size());
                      });

            strategy_gened_ = false;
            checked_ = false;
        }


        // 0: All parameter are on Device
        //max: sizeof(attrs_), all parameter are on Host
        std::vector<WeightAttr*> gen_strategy(){
                init_strategy();
            strategy_gened_ = true;
                if (meta_.layer_num_ <=2){
                    std::cout << "layer num smaller than 2, cpu_offload will not be adopted" << std::endl;
                    return attrs_;
                }
                int max_level = attrs_.size();
                for (int level = 0; level < max_level; level++){
                    if (device_mem_compute() < capacity()){
                        break;
                    } else{
                        increase_strategy_level();
                    }
                }
            return  attrs_;
        }

//        template<class OP,typename  ...Args>
//        torch::Tensor forward(OP op, const int& cur_layer, Args&&... args){
//            if(!strategy_gened_){
//                std::cerr << "warning : Please invoke gen_strategy of AllocationStrategy before use the forward function" << std::endl;
//            }
//            send_async(cur_layer);
//            op.forward(std::forward<Args>(args)...);
//        }

        torch::Tensor forward(op::MaskedMultiHeadAttention* attn,
                              const int& cur_layer,
                              torch::Tensor& input,
                              const torch::Tensor& pre_padding_length,
                              bool pre_layernorm,
                              bool add_residual,
                              bool first_pass){
            if(!strategy_gened_){
                std::cerr << "warning : Please invoke gen_strategy of AllocationStrategy before use the forward function" << std::endl;
            }
            check_tensor_size();
            send_async(cur_layer);
            torch::Tensor res = attn->forward(input, pre_padding_length, pre_layernorm, add_residual, first_pass);
            return res;
        }

        torch::Tensor forward(op::FeedForwardNetwork* ffn,
                              const int& cur_layer,
                              torch::Tensor& input,
                              bool pre_layernorm,
                              bool add_residual){
            if(!strategy_gened_){
                std::cerr << "warning : Please invoke gen_strategy of AllocationStrategy before use the forward function" << std::endl;
            }
            torch::Tensor res = ffn->forward(input, pre_layernorm, add_residual);
            return res;
        }


        //return the tensor on GPU
        torch::Tensor allocate(const std::string& weight_name,const torch::Tensor& h_weight){
            if(!strategy_gened_){
                std::cerr << "warning : Please invoke gen_strategy of AllocationStrategy before use the allocate function" << std::endl;
            }
            assert(h_weight.device() == torch::kCPU && "Give a host tensor for AllocationStrategy's allocate function");
            for (auto iter = attrs_.begin(); iter < attrs_.end(); iter++){
             if(weight_name == wtype2string((*iter)->type_)){
                 int cur_layer = std::max((*iter)->host_tensors.size(),(*iter)->device_tensors.size());
                 //push the tensor to vector for arrangement
                 if((*iter)->on_device_){
                     (*iter)->device_tensors.push_back(h_weight.cuda());
                 }else{
                     std::cout << "going to use pinned memory" << std::endl;
                     (*iter)->host_tensors.push_back(h_weight.pin_memory());
                     std::cout << "finish to use pinned memory" << std::endl;
                     //(*iter)->host_tensors.push_back(h_weight);
                     if (cur_layer == 0 || cur_layer == 1){
                         (*iter)->device_tensors.push_back(h_weight.cuda()); //for swap area
                     }
                 }
                 //return the weight tensor on GPU
                 if(string2wtype(weight_name) == WordEmbedding){
                     assert((*iter)->device_tensors.size() == 1 && (*iter)->host_tensors.size() == 0);
                     return (*iter)->device_tensors.back();
                 }else{
                     if((*iter)->on_device_){
                         return (*iter)->device_tensors.back();
                     }else {
                         if (cur_layer % 2 == 0) {
                             return (*iter)->device_tensors.front();
                         } else {
                             return (*iter)->device_tensors.back();
                         }
                     }
                 }
             }
            }
            //exception
            std::cerr << "illegal weight name !" << std::endl;
            std::cerr << "candidate weight name are : " << std::endl;
            for (auto iter = attrs_.begin(); iter < attrs_.end(); iter++) {
                std::cerr << "   " << wtype2string((*iter)->type_) << "    ";
            }
            std::cout << std::endl;
            return h_weight;
        }

        void print_strategy(){
            mr_.print();
            std::cout << "Memory allocation strategy : " << std::endl;
           for (auto iter = attrs_.begin(); iter < attrs_.end();iter++){
               if ((*iter)->on_device_){
                   std::cout << std::right << std::setw(20) <<
                                wtype2string((*iter)->type_) << " ========== device" << std::endl;
               }else{
                   std::cout << std::right << std::setw(20) <<
                                wtype2string((*iter)->type_) << " ========== host" << std::endl;
               }
           }
        }

        void set_memory_fraction(const float& thres){
            if (thres > 1.0 || thres < 0.0){
                std::cout << "invalid threshold[0.0, 1.0]." << std::endl;
                return;
            }
            thres_ = thres;
        }

        ~AllocationStrategy(){
            for (auto iter = attrs_.begin(); iter < attrs_.end(); iter++){
                delete (*iter);
                *iter = nullptr;
            }
        }


    private:
        void check_tensor_size() {
            if (!checked_) {
                for (auto iter = attrs_.begin(); iter < attrs_.end(); iter++) {
                    if ((*iter)->type_ != WordEmbedding) {
                        if ((*iter)->on_device_) {
                            assert((*iter)->device_tensors.size() == meta_.layer_num_);
                        } else {
                            if (meta_.layer_num_ >= 2) {
                                assert((*iter)->device_tensors.size() == 2);
                            }
                            assert((*iter)->host_tensors.size() == meta_.layer_num_);
                        }
                    } else {
                        assert((*iter)->device_tensors.size() == 1);
                        assert((*iter)->host_tensors.size() == 0);
                    }
                }
                checked_ = true;
            }
        }

        void send_async(const int& cur_layer){
            int next_layer = cur_layer+1;
            if (next_layer >= meta_.layer_num_){
                next_layer = 0;
            }
            for (auto iter = attrs_.begin(); iter < attrs_.end(); iter++){
                if (!(*iter)->on_device_){
                    if(next_layer % 2 == 0){
                        check_cuda_error(
                        cudaMemcpyAsync((*iter)->device_tensors.front().data_ptr(),
                                        (*iter)->host_tensors[next_layer].data_ptr(),
                                        (*iter)->host_tensors[next_layer].nbytes(),
                                        cudaMemcpyHostToDevice,
                                        (*iter)->stream_)
                        );
                    }else{
                        check_cuda_error(
                                cudaMemcpyAsync((*iter)->device_tensors.back().data_ptr(),
                                        (*iter)->host_tensors[next_layer].data_ptr(),
                                        (*iter)->host_tensors[next_layer].nbytes(),
                                        cudaMemcpyHostToDevice,
                                        (*iter)->stream_)
                        );
                    }
                }
            }
        }

        void increase_strategy_level(){
            for (auto iter = attrs_.begin(); iter < attrs_.end(); iter++){
                if (!(*iter)->fixed_ && (*iter)->on_device_) {
                    (*iter)->on_device_ = false;
                    break;
                }
            }
        }

        // set strategy to level 0
        void init_strategy(){
            for (auto iter = attrs_.begin(); iter < attrs_.end(); iter++){
                if (!(*iter)->fixed_) {
                    (*iter)->on_device_ = true;
                }
            }
        }

        size_t device_mem_compute(){
            size_t total = 0;
            for (const auto& attr : attrs_){
                if (attr->type_ == WordEmbedding){
                    total += attr->size() * get_itemsize(meta_);
                }else{
                    if (attr->on_device_){
                        total += attr->size() * meta_.layer_num_ * get_itemsize(meta_) ;
                    }else{
                        total += attr->size() * 2 * get_itemsize(meta_);
                    }
                }
            }
            std::cout << "device mem compute : " << total << std::endl;
            return total;
        }

        float capacity(){
            size_t avail;
            size_t total;
            check_cuda_error(cudaMemGetInfo( &avail, &total ));
            /*
            std::cout << "Device memory available: " << avail << std::endl;
            std::cout << "Total memory  used: " << total << std::endl;
            */
            size_t buffer_size = mr_.get_total_buffer();
            size_t cache_size  = mr_.get_total_cache();
            std::cout << "avail is " << avail << std::endl;
            std::cout << "total is " << total << std::endl;
            float cap =  total * thres_  - buffer_size - cache_size;
            if (cap < 0){
                cap = 0;
            }
            std::cout << "capacity is " << cap << std::endl;
            return cap;
        }


        void register_weight(WeightAttr* p_attr){
            for (auto iter = attrs_.begin(); iter < attrs_.end(); iter++) {
                if (p_attr->type_ == (*iter)->type_){
                    std::cout << "This attribute has already been registered" << std::endl;
                    return;
                }
            }
            attrs_.push_back(p_attr);
        }
        std::vector<WeightAttr*> attrs_;
        MetaDesc meta_;
        float thres_;
        bool strategy_gened_;
        MemReporter mr_;
        bool checked_;
    };
}


#endif //EET_MEM_STRATEGY_H
