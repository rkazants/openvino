// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API RemoveFilteringBoxesBySize;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::RemoveFilteringBoxesBySize: public ngraph::pass::GraphRewrite {
public:
    RemoveFilteringBoxesBySize() : GraphRewrite() {
        remove_filtering_boxes_by_size();
    }

private:
    void remove_filtering_boxes_by_size();
};
