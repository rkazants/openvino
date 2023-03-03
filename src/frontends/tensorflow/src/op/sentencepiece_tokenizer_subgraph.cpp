// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/frontend/tensorflow/decoder.hpp"
#include "openvino/op/sentencepiece_tokenizer.hpp"
#include "tf_framework_node.hpp"

using namespace std;
using namespace ov::opset10;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

namespace {
std::shared_ptr<DecoderBase> extract_decoder(const std::shared_ptr<Node>& node) {
    auto fw_node = std::dynamic_pointer_cast<tensorflow::FrameworkNode>(node);
    FRONT_END_GENERAL_CHECK(fw_node, "The operation node is not FrameworkNode");
    return fw_node->get_decoder();
}
}  // namespace

OutputVector translate_sentencepiece_tokenizer_subgraph(const NodeContext& node) {
    // this is custom translator that converts a sub-graph with SentencePieceOp, SentencePieceTokenizer,
    // and RaggedTensorToSparse operation- into a custom operation SentencepieceTokenizerExtensionOp
    FRONT_END_GENERAL_CHECK(node.get_input_size() > 0, "RaggedTensorToSparse expects at least one input.");

    // check that producers of RaggedTensorToSparse are SentencePieceOp, SentencePieceTokenizer
    auto sp_tokenize_op = node.get_input(0).get_node_shared_ptr();
    auto sp_tokenize_decoder = extract_decoder(sp_tokenize_op);
    FRONT_END_GENERAL_CHECK(sp_tokenize_decoder->get_op_type() == "SentencepieceTokenizeOp",
                            "The translator is not applicable");
    FRONT_END_GENERAL_CHECK(sp_tokenize_op->get_input_size() > 6,
                            "SentencepieceTokenizeOp expects at least six inputs");

    auto sp_op = sp_tokenize_op->input_value(0).get_node_shared_ptr();
    auto sp_decoder = extract_decoder(sp_op);
    FRONT_END_GENERAL_CHECK(sp_decoder->get_op_type() == "SentencepieceOp", "The translator is not applicable");

    // prepare inputs that go to custom operation
    // prepare input 0 - SentencePieceTokenizer configuration model
    auto sp_model_ov_any = sp_decoder->get_attribute("model");
    FRONT_END_GENERAL_CHECK(sp_model_ov_any.is<std::string>(),
                            "SentencePieceTokenizer configuration model is incorrect format");
    auto str_spm_model = sp_model_ov_any.as<std::string>();
    auto sp_model_const = make_shared<Constant>(element::u8, Shape{str_spm_model.size()}, str_spm_model.data());

    // prepare input six inputs
    auto inputs = sp_tokenize_op->input_value(1);
    auto nbest_size = sp_tokenize_op->input_value(2);
    auto alpha = sp_tokenize_op->input_value(3);
    auto add_bos = sp_tokenize_op->input_value(4);
    auto add_eos = sp_tokenize_op->input_value(5);
    auto reverse = sp_tokenize_op->input_value(6);

    OutputVector inputs_vector = OutputVector{sp_model_const, inputs, nbest_size, alpha, add_bos, add_eos, reverse};

    // create a node with custom operation
    auto sp_tokenizer_ext = make_shared<SentencepieceTokenizerExtensionOp>(inputs_vector);

    return sp_tokenizer_ext->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
