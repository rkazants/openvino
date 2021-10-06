// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {
ngraph::OutputVector TranslateRelu6Op(const NodeContext& node) {
    auto data = node.get_ng_input(0);
    auto clamp = std::make_shared<Clamp>(data, 0.0, 6.0f);
    clamp->set_friendly_name(node.get_name());
    return clamp->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
