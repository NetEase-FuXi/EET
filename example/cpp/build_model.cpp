//
// Created by ligz on 2021/4/22.
//

#include "../../csrc/op/op.h"
using namespace eet;

#include "../../csrc/cpu_offload/mem_reporter.h"
#include "../../csrc/cpu_offload/mem_strategy.h"


std::vector<torch::Tensor> make_embedding_weights(MetaDesc meta, int vocab_size){
    torch::TensorOptions options = torch::TensorOptions().dtype(meta.dtype_).device(torch::kCPU).requires_grad(false);
    torch::Tensor embedding_weights =
            torch::ones({vocab_size, meta.hidden_units_}, options);
    torch::Tensor position_weights  =
            torch::empty({0}, options);
    torch::Tensor token_type_weights =
            torch::empty_like(position_weights);
    torch::Tensor ln_weights = torch::empty_like(position_weights);
    torch::Tensor ln_bias = torch::empty_like(position_weights);
    std::vector<torch::Tensor> tmp{embedding_weights,
                                   position_weights,
                                   token_type_weights,
                                   ln_weights,ln_bias};
    return tmp;
}

std::vector<torch::Tensor> make_attn_weights(MetaDesc meta){
    torch::TensorOptions options = torch::TensorOptions().dtype(meta.dtype_).device(torch::kCPU).requires_grad(false);
    torch::Tensor q_weights =
            torch::ones({meta.hidden_units_,meta.hidden_units_},options);
    torch::Tensor q_bias    =
            torch::ones({meta.hidden_units_},options);
    torch::Tensor k_weights =
            torch::ones({meta.hidden_units_,meta.hidden_units_},options);
    torch::Tensor k_bias    =
            torch::ones({meta.hidden_units_},options);
    torch::Tensor v_weights =
            torch::ones({meta.hidden_units_,meta.hidden_units_},options);
    torch::Tensor v_bias    =
            torch::ones({meta.hidden_units_},options);
    torch::Tensor proj_weights =
            torch::ones({meta.hidden_units_,meta.hidden_units_},options);
    torch::Tensor proj_bias    =
            torch::ones({meta.hidden_units_},options);
    torch::Tensor ln_weights =
            torch::ones({meta.hidden_units_}, options);
    torch::Tensor ln_bias =
            torch::ones({meta.hidden_units_}, options);
    std::vector<torch::Tensor> tmp{q_weights,
                                   k_weights,
                                   v_weights,
                                   q_bias,
                                   k_bias,
                                   v_bias,
                                   proj_weights,
                                   proj_bias,
                                   ln_weights,
                                   ln_bias};
    return tmp;
}

std::vector<torch::Tensor> make_ffn_weights(MetaDesc meta){
    torch::TensorOptions options = torch::TensorOptions().dtype(meta.dtype_).device(torch::kCPU).requires_grad(false);
    torch::Tensor ffn1_weights =
            torch::ones({meta.hidden_units_, meta.hidden_units_ * 4}, options);
    torch::Tensor ffn1_bias =
            torch::ones({meta.hidden_units_ * 4}, options);
    torch::Tensor ffn2_weights =
            torch::ones({meta.hidden_units_ * 4, meta.hidden_units_}, options);
    torch::Tensor ffn2_bias =
            torch::ones({meta.hidden_units_}, options);
    torch::Tensor ln_weights =
            torch::ones({meta.hidden_units_}, options);
    torch::Tensor ln_bias =
            torch::ones({meta.hidden_units_}, options);
    std::vector<torch::Tensor> tmp{ffn1_weights,
                                   ffn1_bias,
                                   ffn2_weights,
                                   ffn2_bias,
                                   ln_weights,
                                   ln_bias};
    return tmp;

}

class GPT{
public:
    GPT(const MetaDesc& meta, const int& voc_size, std::vector<torch::Tensor> embedding_weights,
        std::vector<std::vector<torch::Tensor>> attn_weights_vec,
        std::vector<std::vector<torch::Tensor>> ffn_weights_vec):meta_(meta),as(meta,voc_size){
        as.set_memory_fraction(0.95);
        as.gen_strategy();
        as.print_strategy();
        embedding_ = std::make_shared<op::Embedding>(meta, as.allocate("word_embedding",embedding_weights[0]),
                                                    embedding_weights[1].cuda(),
                                                    embedding_weights[2].cuda(),
                                                    embedding_weights[3].cuda(),
                                                    embedding_weights[4].cuda());
        assert(attn_weights_vec.size() == meta_.layer_num_);
        assert(ffn_weights_vec.size() == meta_.layer_num_);
        for(auto iter = attn_weights_vec.begin(); iter < attn_weights_vec.end(); iter++){
            op::MaskedMultiHeadAttention attn = op::MaskedMultiHeadAttention(meta,
                                                    as.allocate("attn_q_weights",(*iter)[0]),
                                                    as.allocate("attn_k_weights",(*iter)[1]),
                                                    as.allocate("attn_v_weights",(*iter)[2]),
                                                    as.allocate("attn_q_bias",(*iter)[3]),
                                                    as.allocate("attn_k_bias",(*iter)[4]),
                                                    as.allocate("attn_v_bias",(*iter)[5]),
                                                    as.allocate("attn_output_weights",(*iter)[6]),
                                                    as.allocate("attn_output_bias",(*iter)[7]),
                                                    as.allocate("attn_ln_weights",(*iter)[8]),
                                                    as.allocate("attn_ln_bias",(*iter)[9]));
            attns_.push_back(attn);
        }
        for(auto iter = ffn_weights_vec.begin(); iter < ffn_weights_vec.end(); iter++){
            op::FeedForwardNetwork ffn = op::FeedForwardNetwork(meta,
                                          as.allocate("ffn_inter_weights",(*iter)[0]),
                                          as.allocate("ffn_inter_bias",(*iter)[1]),
                                          as.allocate("ffn_out_weights",(*iter)[2]),
                                          as.allocate("ffn_out_bias",(*iter)[3]),
                                          as.allocate("ffn_ln_weights",(*iter)[4]),
                                          as.allocate("ffn_ln_bias",(*iter)[5]));
            ffns_.push_back(ffn);
        }
    }
    torch::Tensor forward(torch::Tensor& input, const torch::Tensor& pre_padding_length, bool first_pass){
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA).requires_grad(false);
        torch::Tensor positions = torch::arange(1, input.sizes()[1]+1, options);
        int size = positions.sizes()[0];
        positions.repeat({meta_.batch_size_ , size});
        //positions = positions.view((meta_.batch_size_, size));
        torch::Tensor out = embedding_->forward_fairseq(input, positions, false, 1);
        for (int i = 0 ; i < meta_.layer_num_; i++){
            out = attns_[i].forward(out,pre_padding_length, true, true, first_pass);
            out = ffns_[i].forward(out, true, true);
        }
        cudaDeviceSynchronize();
        return out;
    }

    torch::Tensor forward_co(torch::Tensor& input, const torch::Tensor& pre_padding_length, bool first_pass){
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA).requires_grad(false);
        torch::Tensor positions = torch::arange(1, input.sizes()[1]+1, options);
        int size = positions.sizes()[0];
        positions.repeat({meta_.batch_size_ , size});
        //positions = positions.view((meta_.batch_size_, size));
        torch::Tensor out = embedding_->forward_fairseq(input, positions, false, 1);
        for (int i = 0 ; i < meta_.layer_num_; i++){
            out = as.forward(&attns_[i],i,out,pre_padding_length, true, true, first_pass);
            out = as.forward(&ffns_[i],i,out, true, true);
        }
        cudaDeviceSynchronize();
        return out;
    }

private:
    MetaDesc meta_;
    std::shared_ptr<op::Embedding> embedding_;
    std::vector<op::MaskedMultiHeadAttention> attns_;
    std::vector<op::FeedForwardNetwork> ffns_;
    co::AllocationStrategy as;
};

int t_main(c10::ScalarType dtype, int bh,int head, int hidden, int nlayers){
    int batch = bh;
    int head_num = head;
    int hidden_units = hidden;
    int layer_num = nlayers;
    int max_seq_len = 1024;
    int max_full_seq_len = max_seq_len / 2;
    int vocab_size = 10000;
    int full_seq_len = max_full_seq_len;
    int loop_time = 2;
    c10::ScalarType type = dtype;
    MetaDesc meta(batch,head_num,hidden_units,layer_num,
                  max_seq_len,max_full_seq_len,type);


    //construct weights
    std::vector<torch::Tensor> embedding_weights = make_embedding_weights(meta, vocab_size);
    std::vector<std::vector<torch::Tensor>> attn_weights_vec;
    std::vector<std::vector<torch::Tensor>> ffn_weights_vec;
    for(int i = 0; i < meta.layer_num_; i++) {
        std::vector<torch::Tensor> attn_weights = make_attn_weights(meta);
        attn_weights_vec.push_back(attn_weights);
        std::vector<torch::Tensor> ffn_weights = make_ffn_weights(meta);
        ffn_weights_vec.push_back(ffn_weights);
    }
    std::cout << "construct weights finish" << std::endl;

    //construct model
    GPT model(meta, vocab_size, embedding_weights, attn_weights_vec, ffn_weights_vec);
    std::cout << "construct model finish" << std::endl;

    //make the input
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA).requires_grad(false);
    torch::Tensor input_full = torch::randint(vocab_size,{batch,full_seq_len},options);
    torch::Tensor input_inc  = torch::randint(vocab_size,{batch, 1}, options);
    torch::Tensor pre_padding_length = torch::randint(full_seq_len,{meta.batch_size_,1}, options);
    std::cout << "make input finish" << std::endl;
    //inference
    for(int i = 0; i < loop_time; i++){
        std::cout << "loop : " << i << std::endl;
        auto t0 = std::chrono::high_resolution_clock::now();
        torch::Tensor output_full = model.forward_co(input_full, pre_padding_length, true);
        std::cout << "finish full pass" << std::endl;
        auto t11 = std::chrono::high_resolution_clock::now();
        for (int t = full_seq_len; t < max_seq_len; t++) {
            torch::Tensor output_inc = model.forward_co(input_inc, pre_padding_length, false);
        }
        auto t22 = std::chrono::high_resolution_clock::now();
        auto time1 = std::chrono::duration_cast<std::chrono::duration<double>>(t22 - t11).count();
        std::cout << "inc time : " <<
                  time1 / (max_seq_len/2) << "seconds" << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        std::cout << "total time time for " << max_seq_len << " is " <<
                                                                time << "seconds" << std::endl;
    }
    //MManager::get_instance().report_buffer();
    //MManager::get_instance().report_cache();
    MManager::get_instance().clear();
    std::cout << "finish test" << std::endl;
    return 0;
}

int main(){
	int head[9] = {12, 16,16, 32, 32, 32, 32, 40, 96};
	int hidden[9] ={768, 1024, 1536, 2048, 2560, 3072, 4096, 5120, 12288};
	int layer_num[9] = {12, 24, 24, 24, 32, 45, 40, 36, 40};
	int batch[6] = {/*1,*/ 4, /*8,*/ 16, 32, 64};
	
    /*
	for(int i = 3 ; i < 4; i++){
	    int bh = batch[i];
	    for (int j = 3 ; j < 4; j++){
	    	int head_num = head[j];
		int hidden_units = hidden[j];
		std::cout << "*******dtype = Float32 " <<  "********* batch = " << bh << " ********* hidden_units = " << hidden_units <<" *********" << std::endl;
		t_main(torch::kFloat32, bh, head_num, hidden_units);
	    }
	}
	*/
	for(int i = 1 ; i < 2; i++){
	    int bh = batch[i];
	    for (int j = 6 ; j < 7; j++){
	    	int head_num = head[j];
		    int hidden_units = hidden[j];
		    int nlayers = layer_num[j];
		    std::cout << "*******dtype = Float16 " <<  "********* batch = " << bh <<
		            " ********* hidden_units = " << hidden_units <<" ********* layers = " << nlayers <<  "  ***********" << std::endl;
		    t_main(torch::kFloat16, bh, head_num, hidden_units, nlayers);
	    }
	}

	return 0;
}
