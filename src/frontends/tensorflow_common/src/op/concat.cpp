// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov;
using namespace opset8;
using namespace ov::frontend;
using namespace ov::frontend::tensorflow;

#define OPENVINO_CUSTOM_INT64_HACK

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_concat_op(const NodeContext& node) {
    // The difference between Concat and ConcatV2 is that
    // axis is the first input for Concat
    // and is the last input to ConcatV2
    default_op_checks(node, 2, {"Concat", "ConcatV2"});
    auto input_size = static_cast<int>(node.get_input_size());
    int64_t axis;
    ov::Rank input_rank = ov::Rank::dynamic();
    OutputVector inputs;
    if (node.get_op_type() == "Concat") {
        std::vector<int64_t> axis_vector;
        get_const_input(node, 0, &axis_vector);
        TENSORFLOW_OP_VALIDATION(
            node,
            axis_vector.size() == 1,
            "Input model is incorrect: axis input for Concat operation must have exactly one element.");
        axis = axis_vector[0];
        for (int input_idx = 1; input_idx < input_size; ++input_idx) {
            if (node.get_input(input_idx).get_partial_shape().rank().is_static()) {
                input_rank = node.get_input(input_idx).get_partial_shape().rank();
            }
#ifdef OPENVINO_CUSTOM_INT64_HACK
            inputs.push_back(
                input_idx == 2
                    ? std::make_shared<ConvertLike>(node.get_input(input_idx), node.get_input(input_idx - 1))->output(0)
                    : node.get_input(input_idx));
#else
            inputs.push_back(node.get_input(input_idx));
#endif
        }
    } else if (node.get_op_type() == "ConcatV2") {
        std::vector<int64_t> axis_vector;
        get_const_input(node, input_size - 1, &axis_vector);
        TENSORFLOW_OP_VALIDATION(
            node,
            axis_vector.size() == 1,
            "Input model is incorrect: axis input for Concat operation must have exactly one element.");
        axis = axis_vector[0];
        for (int input_idx = 0; input_idx < input_size - 1; ++input_idx) {
            if (node.get_input(input_idx).get_partial_shape().rank().is_static()) {
                input_rank = node.get_input(input_idx).get_partial_shape().rank();
            }
#ifdef OPENVINO_CUSTOM_INT64_HACK
            inputs.push_back(
                input_idx == 1
                    ? std::make_shared<ConvertLike>(node.get_input(input_idx), node.get_input(input_idx - 1))->output(0)
                    : node.get_input(input_idx));
#else
            inputs.push_back(node.get_input(input_idx));
#endif
        }
    } else {
        TENSORFLOW_OP_VALIDATION(node,
                                 false,
                                 "Internal TensorFlow Frontend error: incorrect operation type is passed to "
                                 "translate_concat_op function.");
    }

    // normalize axis to be non-negative
    // this workaround is needed while MO IR Engine is used
    if (input_rank.is_static() && axis < 0) {
        axis += input_rank.get_length();
    }

    auto concat = make_shared<Concat>(inputs, axis);
    set_node_name(node.get_name(), concat);
    return {concat};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
