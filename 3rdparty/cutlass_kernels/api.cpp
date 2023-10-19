#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "fpA_intB_gemm_wrapper.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("fpA_intB_gemm", &fpA_intB_gemm_forward_cuda, "Weight only gemm");
  m.def("preprocess_weights", &preprocess_weights_cuda, "transform_int8_weights_for_cutlass",
        py::arg("origin_weight"),
        py::arg("is_int4") = false,
        py::arg("arch") = 86);
}