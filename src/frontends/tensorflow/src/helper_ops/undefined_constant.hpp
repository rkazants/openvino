// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "internal_operation.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class UndefinedConstant : public InternalOperation {
public:
    OPENVINO_OP("UndefinedConstant", "ov::frontend::tensorflow::util", InternalOperation);

    UndefinedConstant(const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder, {}, 1) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, ov::element::undefined, ov::PartialShape::dynamic());
    }
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
