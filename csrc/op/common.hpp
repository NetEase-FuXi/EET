#ifndef _OP_COMMON_HPP_
#define _OP_COMMON_HPP_

#include <vector>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "opbase.h"

 #define _DEBUG_MODE_
#define QKV_PTR_SIZE 3
#define FUSED_QKV_PTR_SIZE 9

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorString(error);
}

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    throw std::runtime_error(std::string("[EET][ERROR] CUDA runtime error: ") + \
        (_cudaGetErrorEnum(result)) + " " + file +  \
        ":" + std::to_string(line) + " \n");\
  }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)

#define RUN_KERNEL(FUNCTION,DTYPE,...)                        \
  if (DTYPE == torch::kFloat32) {                             \
    FUNCTION<float>(__VA_ARGS__);                             \
  } else {                                                    \
    FUNCTION<half>(__VA_ARGS__);                              \
  }                                                           \

#endif