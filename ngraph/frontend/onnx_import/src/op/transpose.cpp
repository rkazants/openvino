//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <memory>
#include <vector>

#include "ngraph/builder/reshape.hpp"
#include "ngraph/node.hpp"
#include "onnx_import/op/transpose.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector transpose(const Node& node)
                {
                    Output<ngraph::Node> data = node.get_ng_inputs().at(0);

                    auto permute_axes =
                        node.get_attribute_value<std::vector<std::size_t>>("perm", {});

                    return {(permute_axes.empty())
                                ? ngraph::builder::opset1::transpose(data)
                                : ngraph::builder::opset1::reorder_axes(data, permute_axes)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
