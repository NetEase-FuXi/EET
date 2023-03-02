#include <torch/extension.h>
#include "op/ffn.hpp"
#include "op/t5ffn.hpp"
#include "op/embedding.hpp"
#include "op/layer_norm.hpp"
#include "op/multi_head_attention.hpp"
#include "op/cross_multi_head_attention.hpp"
#include "op/masked_multi_head_attention.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    #ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
        m.attr("__version__") = "dev";
    #endif

    py::class_<eet::MetaDesc>(m, "MetaDesc")
        .def(py::init<const int&, const int&,const int&, const int&, const int&, const int&, const int&,const int&, const py::object&, const std::string& ,const bool&, const std::string& >());

    py::class_<eet::op::MaskedMultiHeadAttention>(m, "MaskedMultiHeadAttention")
        .def(py::init<eet::MetaDesc,const torch::Tensor&,const torch::Tensor&,
                                const torch::Tensor&,const torch::Tensor&,
                                const torch::Tensor&,const torch::Tensor&,
                                const torch::Tensor&,const torch::Tensor&>())
        .def("forward", &eet::op::MaskedMultiHeadAttention::forward, "MaskedMultiHeadAttention forward", py::arg("input"), py::arg("pre_padding_len"), py::arg("reorder_state"), py::arg("pre_layernorm"), py::arg("add_residual"), py::arg("first_pass"), py::arg("relative_attention_bias")=torch::empty(0));
    
    py::class_<eet::op::CrossMultiHeadAttention>(m, "CrossMultiHeadAttention")
        .def(py::init<eet::MetaDesc,const torch::Tensor& ,const torch::Tensor&,
                                const torch::Tensor& ,const torch::Tensor&,
                                const torch::Tensor&,const torch::Tensor&,
                                const torch::Tensor&,const torch::Tensor&,
                                const torch::Tensor&,const torch::Tensor&>())
        .def("forward", &eet::op::CrossMultiHeadAttention::forward, "CrossMultiHeadAttention forward");

    py::class_<eet::op::MultiHeadAttention>(m, "MultiHeadAttention")
    .def(py::init<eet::MetaDesc,const torch::Tensor& ,const torch::Tensor&,
                            const torch::Tensor&,const torch::Tensor&,
                            const torch::Tensor&,const torch::Tensor&,
                            const torch::Tensor&,const torch::Tensor&>())
    .def("forward", &eet::op::MultiHeadAttention::forward, "MultiHeadAttention forward", py::arg("input"), py::arg("padding_mask"), py::arg("pre_layernorm"), py::arg("add_residual"), py::arg("need_sequence_mask")=false, py::arg("relative_attention_bias")=torch::empty(0));

    py::class_<eet::op::FeedForwardNetwork>(m, "FeedForwardNetwork")
        .def(py::init<eet::MetaDesc,
            const torch::Tensor&,const torch::Tensor&,
            const torch::Tensor&,const torch::Tensor&,
            const torch::Tensor&,const torch::Tensor&,
            const std::string&>())
        .def("forward", &eet::op::FeedForwardNetwork::forward, "FeedForwardNetwork forward");

    py::class_<eet::op::T5FeedForwardNetwork>(m, "T5FeedForwardNetwork")
        .def(py::init<eet::MetaDesc,
            const torch::Tensor&,const torch::Tensor&,
            const torch::Tensor&,const torch::Tensor&,
            const torch::Tensor&,const torch::Tensor&,
            const torch::Tensor&,const torch::Tensor&,
            const std::string&>())
        .def("forward", &eet::op::T5FeedForwardNetwork::forward, "T5FeedForwardNetwork forward");

    py::class_<eet::op::Embedding>(m, "Embedding")
        .def(py::init<eet::MetaDesc, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const std::string&>())
        .def("forward_fairseq", &eet::op::Embedding::forward_fairseq, "Embedding forward_fairseq")
        .def("forward_transformers", &eet::op::Embedding::forward_transformers, "Embedding forward_transformers");


    py::class_<eet::op::LayerNorm>(m, "LayerNorm")
        .def(py::init<eet::MetaDesc, const torch::Tensor&, const torch::Tensor&>())
        .def("layer_norm", &eet::op::LayerNorm::layer_norm, "layer_norm");
        // .def("AddBiasLayerNorm", &eet::op::layer_norm::AddBiasLayerNorm, "AddBiasLayerNorm");
}
