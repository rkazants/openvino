// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_space_to_depth_op(const NodeContext& node) {
    default_op_checks(node, 1, {"SpaceToDepth"});
    auto input_data = node.get_input(0);

    // retrieve attributes
    auto block_size = node.get_attribute<int64_t>("block_size");
    auto data_format = node.get_attribute<string>("data_format", "NHWC");

    TENSORFLOW_OP_VALIDATION(node,
                             data_format == "NHWC" || data_format == "NCHW",
                             "TensorFlow Frontend supports input data for SpaceToDepth either in NHWC or NCHW format.");
    bool is_nhwc = (data_format == "NHWC");

    convert_nhwc_to_nchw(is_nhwc, input_data);
    auto mode = SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    auto space_to_depth = make_shared<SpaceToDepth>(input_data, mode, block_size)->output(0);
    convert_nchw_to_nhwc(is_nhwc, space_to_depth);
    set_node_name(node.get_name(), space_to_depth.get_node_shared_ptr());
    return {space_to_depth};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
