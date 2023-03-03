// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <string>
#include <vector>

#include "openvino/core/type/non_tensor_type.hpp"
#include "openvino/opsets/opset10.hpp"

/*
// For some helper structures
#include "sentencepiece_processor.h"
#include "str_ops.hpp"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_join.h"
*/
#include "str_ops.hpp"
#include "sentencepiece_tokenizer_extension.h"

namespace ov {

// This is a temporary extension op that consumes multiple operations from TF graph:
// SentencepieceOp + SentencepieceTokenizeOp + RaggedTensorToSparse
// It supports both structural type Str as a single input and decomposed Str Tensor
// represented as regular 3 OV tensors: indices of begins, indices of ends and
// all strings concatenated as U8 1D tensor
class OPENVINO_API SentencepieceTokenizerExtensionOp : public frontend::tensorflow::StructuralTypedOp {
public:
    OPENVINO_OP("SentencepieceTokenizerExtensionOp", "0", frontend::tensorflow::StructuralTypedOp);

    SentencepieceTokenizerExtensionOp(
        const OutputVector& arguments,
        // TODO: Add necessary attribute parameters or extra constant inputs based on TF graph nodes
        const frontend::tensorflow::StructuralTypeProxy::BindInputs& bind_inputs = {})
        : StructuralTypedOp(arguments, bind_inputs) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        // Handle validation model and evaluatation mode due to CPU bug (see other ops)

        // TODO: Move to cpp file

        if (all_inputs_are_constants(this)) {
            ov::TensorVector inputs;
            for (size_t i = 0; i < get_input_size(); ++i) {
                auto constant = std::dynamic_pointer_cast<ov::opset1::Constant>(get_input_node_shared_ptr(i));
                inputs.push_back(Tensor(constant->get_element_type(),
                                        constant->get_shape(),
                                        const_cast<void*>(constant->get_data_ptr())));
            }
            std::vector<int64_t> m_sparse_indices;
            std::vector<int64_t> m_sparse_values;
            std::vector<int64_t> m_sparse_dense_shape;

            evaluate_helper(inputs, m_sparse_indices, m_sparse_values, m_sparse_dense_shape);
            auto num_values = m_sparse_values.size();
            set_output_type(0, element::i32, Shape{num_values, 2});
            set_output_type(1, element::i32, Shape{num_values});
            set_output_type(2, element::i32, Shape{2});
            return;
        }

        set_output_type(0, element::i32, PartialShape{Dimension(), Dimension(2)});
        set_output_type(1, element::i32, PartialShape{Dimension()});
        set_output_type(2, element::i32, PartialShape{Dimension()});
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return std::make_shared<SentencepieceTokenizerExtensionOp>(
            inputs,
            frontend::tensorflow::StructuralTypeProxy::StructuralTypeMapAttribute::get_input(get_rt_info()));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        // Add necessary attributes if any
        return true;
    }

    bool evaluate_helper(const ov::TensorVector& inputs,
                         std::vector<int64_t>& sparse_indices,
                         std::vector<int64_t>& sparse_values,
                         std::vector<int64_t>& sparse_dense_shape) const {
        // inputs should have at least 3 tensors for input strings
        // [0] i32 tensor of begin indices, indices are offsets in [2]
        // [1] i32 tensor of end indices, indices are offsets in [2]
        // [2] 1D u8 tensor of bytes where all strings are concatenated

        // the operation has the following inputs:
        // 0. spm_model
        // data inputs
        // 1. [0] i32 tensor of begin indices, indices are offsets in [2]
        // 2. [1] i32 tensor of end indices, indices are offsets in [2]
        // 3. [2] 1D u8 tensor of bytes where all strings are concatenated
        // 4. nbest_size
        // 5. alpha
        // 6. add_bos
        // 7. add_eos
        // 8. reverse
        auto spm_model = static_cast<char*>(inputs[0].data());
        auto spm_model_size = inputs[0].get_byte_size();

        auto begin_ids = static_cast<int32_t*>(inputs[1].data());
        auto end_ids = static_cast<int32_t*>(inputs[2].data());
        auto data = static_cast<uint8_t*>(inputs[3].data());
        auto batch_size = inputs[1].get_size();

        auto nbest_size = *static_cast<int32_t*>(inputs[4].data());
        auto alpha = *static_cast<float*>(inputs[5].data());
        auto add_bos = *static_cast<bool*>(inputs[6].data());
        auto add_eos = *static_cast<bool*>(inputs[7].data());
        auto reverse = *static_cast<bool*>(inputs[7].data());
        sparse_dense_shape.clear();
        sparse_indices.clear();
        sparse_values.clear();
        sentencepiece_tokenizer_extension(spm_model,
                                          spm_model_size,
                                          begin_ids,
                                          end_ids,
                                          data,
                                          batch_size,
                                          nbest_size,
                                          alpha,
                                          add_bos,
                                          add_eos,
                                          reverse,
                                          sparse_indices,
                                          sparse_values,
                                          sparse_dense_shape);
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
        std::vector<int64_t> m_sparse_indices;
        std::vector<int64_t> m_sparse_values;
        std::vector<int64_t> m_sparse_dense_shape;

        evaluate_helper(inputs, m_sparse_indices, m_sparse_values, m_sparse_dense_shape);
        std::vector<int32_t> m_sparse_indices_i32(m_sparse_indices.begin(), m_sparse_indices.end());
        std::vector<int32_t> m_sparse_values_i32(m_sparse_values.begin(), m_sparse_values.end());
        std::vector<int32_t> m_sparse_dense_shape_i32(m_sparse_dense_shape.begin(), m_sparse_dense_shape.end());

        memcpy(outputs[0].data(), m_sparse_indices_i32.data(), sizeof(int32_t) * m_sparse_indices_i32.size());
        memcpy(outputs[1].data(), m_sparse_values_i32.data(), sizeof(int32_t) * m_sparse_values_i32.size());
        memcpy(outputs[2].data(), m_sparse_dense_shape_i32.data(), sizeof(int32_t) * m_sparse_dense_shape_i32.size());
        return true;
    }

    bool has_evaluate() const {
        return true;
    }

};

}  // namespace ov
