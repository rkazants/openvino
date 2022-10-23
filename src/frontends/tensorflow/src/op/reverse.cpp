// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset9.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset9;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
shared_ptr<Node> compute_sequence_lengths(const Output<Node>& input_shape, int64_t batch_axis, int64_t seq_axis) {
    auto batch_axis_const = make_shared<Constant>(element::i64, Shape{1}, batch_axis);
    auto seq_axis_const = make_shared<Constant>(element::i64, Shape{1}, seq_axis);
    auto gather_axis = make_shared<Constant>(element::i32, Shape{}, 0);
    auto batch_dim = make_shared<Gather>(input_shape, batch_axis_const, gather_axis);
    auto seq_dim = make_shared<Gather>(input_shape, seq_axis_const, gather_axis);
    auto seq_lengths = make_shared<Broadcast>(seq_dim, batch_dim);

    return seq_lengths;
}

OutputVector translate_reverse_base_op(const NodeContext& node,
                                       const Output<Node>& input,
                                       const std::vector<int64_t>& axes) {
    TENSORFLOW_OP_VALIDATION(
        node,
        axes.size() == 0 || axes.size() == 1,
        "OpenVINO TensorFlow Frontend does not support Reverse and ReverseV2 with multiple axes for the reversing.");
    auto reverse_v2_node_name = node.get_name();

    if (axes.size() == 0) {
        // there is nothing to reverse
        input.get_tensor().add_names({reverse_v2_node_name + ":0"});
        return {input};
    }

    int64_t seq_axis = axes[0];
    int64_t batch_axis = 0;

    // normalize batch_axis and seq_axis so that they become non-negative and different
    std::vector<int64_t> unsqueeze_axes;
    if (seq_axis == 0) {
        // unsqueeze is needed by axis = 0, 1
        unsqueeze_axes.push_back(0);
        unsqueeze_axes.push_back(1);
        seq_axis += 2;
    } else if (seq_axis < 0) {
        // for seq_axis is negative, we can handle it only if input rank is defined
        // since we need it for seq_axis normalization
        auto input_rank = input.get_partial_shape().rank();
        if (input_rank.is_static()) {
            auto input_rank_value = static_cast<int64_t>(input_rank.get_length());
            if (input_rank_value == 0) {
                // in case of a scalar input, there is nothing to do
                // we just skip ReverseV2 node and transfer output tensor name
                input.get_tensor().add_names({reverse_v2_node_name + ":0"});
                return {input};
            } else if (input_rank_value == 1) {
                unsqueeze_axes.push_back(0);
                seq_axis = 1;
            } else {
                // normalize seq_axis
                seq_axis += input_rank_value;
                // batch_axis and seq_axis must be different
                batch_axis = (seq_axis == 0 ? 1 : 0);
            }
        } else {
            // in this case we cannot compute batch_axis because normalized seq_axis is unknown
            TENSORFLOW_OP_VALIDATION(node,
                                     false,
                                     "OpenVINO TensorFlow Frontend does not support negative axis and input of "
                                     "undefined rank simultaneously.");
        }
    }

    auto batched_input = input;
    if (unsqueeze_axes.size() > 0) {
        // prepare input to issue auxiliary dimensions for batch
        auto unsqueeze_axes_const = make_shared<Constant>(element::i64, Shape{unsqueeze_axes.size()}, unsqueeze_axes);
        batched_input = make_shared<Unsqueeze>(input, unsqueeze_axes_const);
    }

    auto input_shape = make_shared<ShapeOf>(batched_input, element::i32);
    auto seq_lenghts = compute_sequence_lengths(input_shape, batch_axis, seq_axis);
    auto reverse_sequence = make_shared<ReverseSequence>(batched_input, seq_lenghts, batch_axis, seq_axis)->output(0);

    if (unsqueeze_axes.size() > 0) {
        // remove earlier added additional dimensions from the result
        auto squeeze_axes_const = make_shared<Constant>(element::i64, Shape{unsqueeze_axes.size()}, unsqueeze_axes);
        reverse_sequence = make_shared<Squeeze>(reverse_sequence, squeeze_axes_const);
    }

    set_node_name(node.get_name(), reverse_sequence.get_node_shared_ptr());
    return {reverse_sequence};
}

OutputVector translate_reverse_op(const NodeContext& node) {
    // The second input of Reverse is a boolean vector.
    // True elements correspond the axes along which
    // elements of the input tensor are reversed
    default_op_checks(node, 2, {"Reverse"});
    auto input = node.get_input(0);

    std::vector<bool> dims;
    get_const_input(node, 1, &dims);

    // collect axes along which to reverse
    std::vector<int64_t> axes;
    for (int64_t ind = 0; ind < dims.size(); ++ind) {
        if (dims[ind]) {
            axes.push_back(ind);
        }
    }
    TENSORFLOW_OP_VALIDATION(node,
                             axes.size() == 1,
                             "OpenVINO TensorFlow Frontend supports only constant axis input with one element.");

    return translate_reverse_base_op(node, input, axes);
}

OutputVector translate_reverse_v2_op(const NodeContext& node) {
    // The second input of ReverseV2 is a vector of axes along which
    // elements of the input tensor are reversed
    default_op_checks(node, 2, {"ReverseV2"});
    auto input = node.get_input(0);

    // the translator is able to convert ReverseV2 only
    // if axis is constant and has one element.
    // this limitation is due to the presence of batch_axis and seq_axis attributes.
    // the current limitation is sufficient for parity with Legacy MO frontend.
    std::vector<int64_t> axes;
    get_const_input(node, 1, &axes);

    return translate_reverse_base_op(node, input, axes);
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
