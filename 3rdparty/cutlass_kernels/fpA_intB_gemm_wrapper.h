#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor>
symmetric_quantize_last_axis_of_tensor(torch::Tensor &weight,
                                       py::object &quant_type,
                                       bool return_unprocessed_quantized_tensor);

torch::Tensor preprocess_weights_cuda(torch::Tensor &ori_weight,
                                      bool is_int4);

void fpA_intB_gemm_forward_cuda(torch::Tensor &input,
                                torch::Tensor &weight,
                                torch::Tensor &scale,
                                torch::Tensor &output,
                                int m, int n, int k);