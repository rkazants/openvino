/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#ifndef NGRAPH_TF_BRIDGE_BUILDER_H_
#define NGRAPH_TF_BRIDGE_BUILDER_H_

#include <ostream>
#include <vector>
#include <string>
#include <iterator>
#include <algorithm>
#include <sstream>

#include "graph.hpp"

// TODO: remove explicit proto dependency from this common header
//#include "graph.pb.h"

#include "node_context_impl.hpp"

#include "ngraph/ngraph.hpp"

namespace tensorflow {

using ngraph::OutputVector;
using ngraph::frontend::tensorflow::detail::NodeContext;

// TODO: Get rid of direct usage of this structures and remove the following usages:
using ngraph::frontend::tensorflow::detail::TFNodeDecoder;
using ngraph::frontend::tensorflow::detail::TensorWrapper;



namespace ngraph_bridge {



#if 0
#define NGRAPH_VLOG(I) std::cerr
#else
#define NGRAPH_VLOG(I) std::ostringstream()
#endif


class Builder {
 public:
    static void TranslateGraph(
        std::shared_ptr<ngraph::frontend::InputModelTF> tf_model,
        const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>& static_input_map,
        const std::string name,
        std::shared_ptr<ngraph::Function>& ng_function);

  using OpMap = std::unordered_map<std::string,
                                   std::vector<ngraph::Output<ngraph::Node>>>;
  using ConstMap = std::map<
      ngraph::element::Type,
      std::pair<std::function<Status(const NodeContext&, ngraph::element::Type,
                                     ngraph::Output<ngraph::Node>&)>,
                const ngraph::element::Type>>;
  static const Builder::ConstMap& TF_NGRAPH_CONST_MAP();

  template <typename T>
  static void MakePadding(const std::string& tf_padding_type,
                          const ngraph::Shape& ng_image_shape,
                          const ngraph::Shape& ng_kernel_shape,
                          const ngraph::Strides& ng_strides,
                          const ngraph::Shape& ng_dilations,
                          T& ng_padding_below, T& ng_padding_above) {
    if (tf_padding_type == "SAME") {
      ngraph::Shape img_shape = {0, 0};
      img_shape.insert(img_shape.end(), ng_image_shape.begin(),
                       ng_image_shape.end());
      ngraph::infer_auto_padding(img_shape, ng_kernel_shape, ng_strides,
                                 ng_dilations, ngraph::op::PadType::SAME_UPPER,
                                 ng_padding_above, ng_padding_below);
    } else if (tf_padding_type == "VALID") {
      ng_padding_below.assign(ng_image_shape.size(), 0);
      ng_padding_above.assign(ng_image_shape.size(), 0);
    }
  }

  // This function is used to trace which ng node came from which tf node
  // It does 3 things:
  // 1. Attaches provenance tags. This is guaranteed to propagate the tag info
  // to all nodes.
  // The next 2 are not guaranteed to be present for all nodes.
  // But when present they are correct and agree with provenance tags
  // 2. Attaches friendly names.
  // 3. Prints a log if NGRAPH_TF_LOG_PLACEMENT=1
  static void SetTracingInfo(const std::string& op_name,
                             const ngraph::Output<ngraph::Node> ng_node);
};

inline std::string StrJoin (const std::vector<std::string>& strs, const char* sep)
{
    std::ostringstream str;
    std::copy(strs.begin(), strs.end(), std::ostream_iterator<std::string>(str, sep));
    return str.str();
}

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif
