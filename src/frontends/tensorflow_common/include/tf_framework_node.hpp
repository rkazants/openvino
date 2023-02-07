// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>

#include "openvino/frontend/decoder.hpp"
#include "openvino/op/util/framework_node.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class FrameworkNode : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("FrameworkNode", "util", ::ov::op::util::FrameworkNode);

    FrameworkNode(const std::shared_ptr<DecoderBase>& decoder, const OutputVector& inputs, size_t num_outputs)
        : ov::op::util::FrameworkNode(inputs, std::max(num_outputs, size_t(1))),
          m_decoder(decoder) {
        ov::op::util::FrameworkNodeAttrs attrs;
        // TODO: are there transformations which rely on correct type name?
        //attrs.set_type_name(m_decoder->get_op_type());
        // Something goes wrong with get_op_type -- it returns bad values, may lead to crash
        attrs.set_type_name("FrameworkNode");
        //attrs["tf_orig_type"] = std::string(m_decoder->get_op_type());
        //get_rt_info()["tf_orig_type"] = std::string(m_decoder->get_op_type());
        //std::cerr << "[ DECODER ] " << m_decoder->get_op_type() << "\n";
        //op_type = m_decoder->get_op_type();

        set_attrs(attrs);

        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        for (size_t i = 0; i < get_output_size(); ++i) {
            set_output_type(i, ov::element::dynamic, PartialShape::dynamic());
        }
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return std::make_shared<FrameworkNode>(m_decoder, inputs, get_output_size());
    }

    std::string get_op_type() const {
        return m_decoder->get_op_type();
    }

    std::shared_ptr<DecoderBase> get_decoder() const {
        return m_decoder;
    }
    //std::string op_type;

private:
    std::shared_ptr<DecoderBase> m_decoder;
};
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
