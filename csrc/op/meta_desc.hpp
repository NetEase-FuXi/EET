#ifndef _METADESC_
#define _METADESC_

#include "op/common.hpp"

namespace eet {

// Metadata description 
class MetaDesc{
    public:
    
    MetaDesc(const int& batch_size,const int& head_num, const int& hidden_units,const int& layer_num, 
                const int& max_seq_len,
                const int& max_full_seq_len,
                const py::object& dtype,
                const std::string& cuda_device = "cuda:0",
                const bool& requires_grad = false,
                const std::string& activation_fn = "gelu"):
            batch_size_(batch_size), 
            head_num_(head_num),
            hidden_units_(hidden_units), 
            layer_num_(layer_num),
            max_seq_len_(max_seq_len),
            max_full_seq_len_(max_full_seq_len),
            activation_fn_(activation_fn)
    {
        dtype_ = torch::python::detail::py_object_to_dtype(dtype);
    
        options_ = torch::TensorOptions().dtype(dtype_).device(cuda_device).requires_grad(requires_grad);
        switch(dtype_){
            case torch::kFloat32:
                computeType_ = CUDA_R_32F;
                break;
            case torch::kFloat16:
                computeType_ = CUDA_R_16F;
                break;
            //TODO    
            case torch::kInt8:
                break;
            default:
                break;
        }
        is_available(); 
        if (cublasHandle == nullptr || stream == nullptr){  
            // printf("create handel\n");
            check_cuda_error(cublasCreate(&cublasHandle));
            check_cuda_error(cudaStreamCreate(&stream));
            check_cuda_error(cublasSetStream(cublasHandle, stream));
        }
    }

    //construct from c++
    MetaDesc(const int& batch_size,const int& head_num, const int& hidden_units,const int& layer_num,
             const int& max_seq_len,
             const int& max_full_seq_len,
             const c10::ScalarType& dtype,
             const std::string& cuda_device = "cuda:0",
             const bool& requires_grad = false,
             const std::string& activation_fn = "gelu"):
            batch_size_(batch_size),
            head_num_(head_num),
            hidden_units_(hidden_units),
            layer_num_(layer_num),
            max_seq_len_(max_seq_len),
            max_full_seq_len_(max_full_seq_len),
            dtype_(dtype),
            activation_fn_(activation_fn)
    {
        options_ = torch::TensorOptions().dtype(dtype_).device(cuda_device).requires_grad(requires_grad);
        switch(dtype_){
            case torch::kFloat32:
                computeType_ = CUDA_R_32F;
                break;
            case torch::kFloat16:
                computeType_ = CUDA_R_16F;
                break;
                //TODO
            case torch::kInt8:
                break;
            default:
                break;
        }
        is_available();
        if (cublasHandle == nullptr || stream == nullptr){
            // printf("create handel\n");
            check_cuda_error(cublasCreate(&cublasHandle));
            check_cuda_error(cudaStreamCreate(&stream));
            check_cuda_error(cublasSetStream(cublasHandle, stream));
        }

    }

    MetaDesc(const MetaDesc& meta) = default;
    MetaDesc& operator=(const MetaDesc& meta) = default;

    int batch_size_;
    int head_num_;
    int hidden_units_;
    int max_seq_len_;
    int max_full_seq_len_;
    int layer_num_;
    std::string activation_fn_;
    torch::TensorOptions options_;
    cudaDataType_t computeType_;
    c10::ScalarType dtype_;

    static cublasHandle_t cublasHandle;
    static cudaStream_t stream;


    void is_available(){
        assert(batch_size_ > 0 && "batch size must > 0");
        assert(head_num_ > 0 && "head_num must > 0");
        assert(hidden_units_ % head_num_ == 0 && " hidden_units must a multiple of head_num");
        assert(layer_num_ > 0 && "layer_num must > 0");
        assert(max_seq_len_ > 0 && "max_seq_len must > 0");
        assert(max_full_seq_len_ > 0 && "max_seq_len must > 0");
        assert((options_.dtype() == torch::kFloat32 || options_.dtype() == torch::kFloat16 ||
        options_.dtype() == torch::kInt8) && "EET now only support float / half / int8" );
        assert(options_.device().is_cuda() && "EET now only support CUDA");
        assert(options_.requires_grad() == false && "EET now only support inference");
    }
  };
}

#endif