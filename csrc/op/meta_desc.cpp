#include "op/meta_desc.hpp"
namespace eet{
    cublasHandle_t MetaDesc::cublasHandle = nullptr;
    cudaStream_t MetaDesc::stream = nullptr;
}