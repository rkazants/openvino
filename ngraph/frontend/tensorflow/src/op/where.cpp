// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

#if 0

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateWhereOp(
    const NodeContext& node) {
  Output<Node> ng_cond;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_cond));
  auto non_zero = ConstructNgNode<NonZero>(node.get_name(), ng_cond);
  auto transpose_order = ConstructNgNode<Constant>(
      node.get_name(), ngraph::element::i64, ngraph::Shape{2},
      std::vector<int64_t>({1, 0}));
  SaveNgOp(ng_op_map, node.get_name(), ConstructNgNode<Transpose>(
                                      node.get_name(), non_zero, transpose_order));
  return Status::OK();
}

}
}
#endif