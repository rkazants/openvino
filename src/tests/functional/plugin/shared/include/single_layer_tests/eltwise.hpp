// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <shared_test_classes/single_layer/eltwise.hpp>

namespace ov {
namespace test {
namespace subgraph {

TEST_P(EltwiseLayerTest, EltwiseTests) {
    run();
}

} // namespace subgraph
} // namespace test
} // namespace ov
