#ifndef _METADESC_
#define _METADESC_

#include "op/common.hpp"

namespace eet {

// Metadata description 
class MetaDesc{
    public:
        MetaDesc(const py::object &dtype,
                 const int &batch_size,
                 const int &head_num,
                 const int &hidden_units,
                 const int &layer_num,
                 const int &max_seq_len,
                 const int &max_full_seq_len,
                 const std::string &activation_fn,
                 const int &d_kv,
                 const int &d_ff,
                 const std::string &cuda_device = "cuda:0",
                 const bool &requires_grad = false,
                 const float &layernorm_eps = 1e-6,
                 const bool &is_int8 = false) : batch_size_(batch_size),
                                                      head_num_(head_num),
                                                      hidden_units_(hidden_units),
                                                      d_kv_(d_kv),
                                                      d_ff_(d_ff),
                                                      layer_num_(layer_num),
                                                      max_seq_len_(max_seq_len),           // prompt seq_len
                                                      max_full_seq_len_(max_full_seq_len), // max generated seq_len
                                                      activation_fn_(activation_fn),
                                                      layernorm_eps_(layernorm_eps),
                                                      is_int8_(is_int8)
        {
            dtype_ = torch::python::detail::py_object_to_dtype(dtype);

            options_ = torch::TensorOptions().dtype(dtype_).device(cuda_device).requires_grad(requires_grad);
            switch (dtype_)
            {
            case torch::kFloat32:
                dataType_ = CUDA_R_32F;
                computeType_ = CUBLAS_COMPUTE_32F_FAST_16F;
                break;
            case torch::kFloat16:
                dataType_ = CUDA_R_16F;
                computeType_ = CUBLAS_COMPUTE_16F;
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
    int d_kv_;
    int d_ff_;
    int max_seq_len_;
    int max_full_seq_len_;
    int layer_num_;
    float layernorm_eps_;
    bool is_int8_;
    std::string activation_fn_;
    torch::TensorOptions options_;
    cudaDataType_t dataType_;           // cuda dtype
    // cudaDataType_t computeType_;     // cuda dtype
    cublasComputeType_t computeType_;   // cublas type
    c10::ScalarType dtype_;             // torch dtype


    static cublasHandle_t cublasHandle;
    static cudaStream_t stream;


    void is_available(){
        assert(batch_size_ > 0 && "batch size must > 0");
        assert(head_num_ > 0 && "head_num must > 0");
        // assert(hidden_units_ % head_num_ == 0 && " hidden_units must a multiple of head_num");
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