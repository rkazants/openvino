// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <default_opset.h>
#include <tensorflow_frontend/node_context.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;


namespace tensorflow {
namespace ngraph_bridge {

OutputVector TranslateSoftmaxOp(const NodeContext& node) {
    auto ng_inp = node.get_ng_input(0);
    auto inp_shape = ng_inp.get_shape();
    size_t rank = inp_shape.size();
    int64_t axes = rank - 1;
    if (rank < 1) {
        throw errors::InvalidArgument("TF Softmax logits must be >=1 dimension");
    }

    return {ConstructNgNode<opset::Softmax>(node.get_name(), ng_inp, axes)};
}

}
}