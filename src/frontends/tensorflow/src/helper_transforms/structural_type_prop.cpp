// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "structural_type_prop.hpp"

#include <memory>
#include <vector>
#include <numeric>

#include "../helper_ops/str_ops.hpp"

#include "openvino/frontend/tensorflow/frontend.hpp"
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "openvino/core/type/non_tensor_type.hpp"
#include <openvino/opsets/opset9.hpp>

using std::make_shared;

namespace {
    // TODO: Remove this duplicate: CPU transforms has it, copied and pasted here

    bool is_data_movement_operation(const std::shared_ptr<ngraph::Node>& node) {
        return ov::is_type<ngraph::op::v0::Squeeze>(node) ||
               ov::is_type<ov::op::v1::StridedSlice>(node) ||
               ov::is_type<ngraph::op::v0::Unsqueeze>(node) ||
               ov::is_type<ngraph::op::v1::Reshape>(node) ||
               ov::is_type<ngraph::op::v1::Transpose>(node) ||
               ov::is_type<ngraph::op::v0::ShuffleChannels>(node) ||
               ov::is_type<ngraph::op::v7::Roll>(node) ||
               ov::is_type<ngraph::op::v0::ReverseSequence>(node) ||
               ov::is_type<ngraph::op::v0::DepthToSpace>(node) ||
               ov::is_type<ngraph::op::v1::BatchToSpace>(node) ||
               ov::is_type<ngraph::op::v1::Broadcast>(node) ||
               ov::is_type<ngraph::op::v3::Broadcast>(node) ||
               ov::is_type<ngraph::op::v1::Gather>(node) ||
               ov::is_type<ngraph::op::v7::Gather>(node) ||
               ov::is_type<ngraph::op::v8::Gather>(node) ||
               ov::is_type<ngraph::op::v0::Parameter>(node);
    }

    bool is_str_operation(const std::shared_ptr<ngraph::Node>& node) {
        return ov::is_type<ov::frontend::tensorflow::CaseFoldUTF8>(node) ||
               ov::is_type<ov::frontend::tensorflow::NormalizeUTF8>(node) ||
               ov::is_type<ov::frontend::tensorflow::StaticRegexReplace>(node);
    }

    bool is_scalar_like(const std::shared_ptr<ngraph::Node>& node) {
        auto constantNode = std::dynamic_pointer_cast<ngraph::opset8::Constant>(node);
        return constantNode != nullptr && shape_size(constantNode->get_shape()) == 1;
    }
} // namespace

namespace ov {
namespace frontend {
namespace tensorflow {
namespace pass {

StructuralTypeProp::StructuralTypeProp() {
    auto data_movement = ngraph::pattern::wrap_type<ov::op::Op>(ov::pass::pattern::op::as_value_predicate(is_data_movement_operation));
    std::cerr << "[ INFO TF FE ] Registering StructuralTypeProp\n";

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();

        std::cerr << "[ INFO TF FE ] Matching data movement op: " << node->get_type_name() << "\n";

        // Depending on operation, propagate structural type field
        // TODO: This code should be moved to the operations themselves, but now we are trying
        // to avoid any impact on OV structures and implement it externally.
        // Code amount required to implement it in core will be similar to what we are doing
        // here except we won't have similar mega-switches based on op types.

        if (auto parameter = std::dynamic_pointer_cast<ov::op::v0::Parameter>(node)) {
            // Parameter should have a special RT info attribute `structural_type` that should be copied
            // to the output tensor rt_info

            std::cerr << "[ INFO TF FE ] Detected Parameter\n";

            StructuralTypeAttribute::copy(parameter->get_rt_info(), parameter->get_output_tensor(0).get_rt_info());
        } else if (auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(node)) {
            std::cerr << "[ INFO TF FE ] Detected Reshape\n";
            StructuralTypeAttribute::copy(reshape->get_input_tensor(0).get_rt_info(), reshape->get_output_tensor(0).get_rt_info());
        }

        return false;
    };

    auto m = make_shared<ov::pass::pattern::Matcher>(data_movement, "ov::frontend::tensorflow::pass::StructuralTypeProp");
    register_matcher(m, callback);
}


ReplaceStrByU81D::ReplaceStrByU81D() {
    auto str_tensor = ngraph::pattern::wrap_type<ov::op::Op>(
        ov::pass::pattern::op::ValuePredicate([](ov::Output<ov::Node> x) {
            std::cerr << "get_rt_info: " << x.get_tensor().get_rt_info().size() << "\n";
            //return false;
            std::cerr.flush();
            auto rank = x.get_tensor().get_partial_shape().rank();
            std::cerr << "[ REPLACE STR BY U81D ] RANK IS " << (rank.is_static() && rank.get_length() == 0) << "\n";
            return rank.is_static() && rank.get_length() == 0 &&
                StructuralTypeAttribute::has_type(x.get_tensor().get_rt_info(), element::StructuralType::Str());
            // FIXME: Check that this is a scalar, otherwise this transformation doesn't work
            // FIXME: For now we re-interpret all tensors that have Str type as a scalar tensors
        }));

    std::cerr << "[ INFO TF FE ] Registering ReplaceStrByU81D\n";

    auto callback = [](ov::pass::pattern::Matcher& m) {
        //return false;
        auto port = m.get_match_value();  // TODO: iterate over all outputs and check each of it to match the criteria
        auto node = m.get_match_root();

        std::cerr << "[ INFO TF FE ] Detected tensor with Str type: " << node->get_type_name() << "\n";

        if (auto parameter = std::dynamic_pointer_cast<ov::op::v0::Parameter>(node)) {
            std::cerr << "Parameter override to u8/1d\n";
            parameter->set_element_type(element::u8);
            parameter->set_partial_shape(PartialShape{Dimension()});
        }

        // Just setting type and shape without shape propagation -- will require re-validation of the function
        // in the end to catch all inconsistencies due to possible bugs.

        port.get_tensor().set_tensor_type(element::u8, PartialShape{Dimension()});
        //std::cerr << "move to original\n";
        //StructuralTypeAttribute::move_to_original(port.get_tensor().get_rt_info());
        return false;
    };

    auto m = make_shared<ov::pass::pattern::Matcher>(str_tensor, "ov::frontend::tensorflow::pass::ReplaceStrByU81D");
    register_matcher(m, callback);
}


class StructPack : public ov::op::Op {
public:
    OPENVINO_OP("INTERNAL::StructPack");

    StructPack(const OutputVector& arguments, Any res_type, const PartialShape& res_shape)
        : ov::op::Op(arguments), m_res_type(res_type), m_res_shape(res_shape) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, element::dynamic, m_res_shape);
        get_output_tensor(0).get_rt_info()["structural_type"] = StructuralTypeAttribute(m_res_type);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return make_shared<StructPack>(inputs, m_res_type, m_res_shape);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        // FIXME: Serialization only, there is no deserialization
        std::string m_res_type_str = m_res_type->to_string();
        visitor.on_attribute("res_type", m_res_type_str);
        visitor.on_attribute("res_shape", m_res_shape);
        return true;
    }

    bool has_evaluate() const {
        return false;
    }

    Any m_res_type;
    PartialShape m_res_shape;
};


bool DecomposeStrParameters::run_on_model(const std::shared_ptr<Model>& model) {
    // Search for Parameter with List[Tensor] types

    ParameterVector parameters = model->get_parameters();
    ParameterVector new_parameters;  // collect decomposed parameters
    for (size_t i = 0; i < parameters.size(); ++i) {
        auto parameter = parameters[i];
        std::cerr << "[ PARAMETER ] " << i << "\n";
        std::cerr << parameter << "\n";

        // Check 1D and Str structural type
        auto rank = parameter->get_partial_shape().rank();
        if(
            rank.is_static() && rank.get_length() == 1 &&
            parameter->get_element_type() == element::dynamic &&
            StructuralTypeAttribute::has_type(parameter->get_rt_info(), element::StructuralType::Str()))
        {
            std::cerr << "Applying decomposition for parameter: " << parameter->get_name() << "\n";
            OutputVector inputs_for_struct_pack;

            // for individual strings start and end indices
            for (size_t i = 0; i < 2; ++i) {
                auto new_parameter =
                    make_shared<opset9::Parameter>(element::i32, parameter->get_partial_shape());
                new_parameters.push_back(new_parameter);
                inputs_for_struct_pack.push_back(new_parameter);
                // TODO: add links via RT info between original parameter and new ones
            }

            // for tensor elements
            auto new_parameter =
                // using u8 here because we know that we are dealing with strings
                make_shared<opset9::Parameter>(/*element::dynamic*/element::u8, PartialShape{Dimension()});
            new_parameters.push_back(new_parameter);
            inputs_for_struct_pack.push_back(new_parameter);

            auto struct_pack = make_shared<StructPack>(
                inputs_for_struct_pack,
                element::StructuralType::Str(), // parameter->get_rt_info()["structural_type"].as<StructuralTypeAttribute>().value
                parameter->get_partial_shape()
            );

            replace_node(parameter, struct_pack);
            model->remove_parameter({parameter});
        }
    }

    model->add_parameters(new_parameters);
    return true;
}


OutputVector get_inputs (std::shared_ptr<Node> node) {
    OutputVector result;
    for(size_t i = 0; i < node->get_input_size(); ++i) {
        result.push_back(node->get_input_source_output(i));
    }
    return result;
}


ThroughStrOpsProp::ThroughStrOpsProp() {
    // Should better match node that has at least one StructPack at least at one inputs
    auto node = ngraph::pattern::wrap_type<ov::op::Op>(
        ov::pass::pattern::op::as_value_predicate(is_str_operation));

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();
        std::cerr << "[ INFO TF FE ] Matching str op: " << node->get_type_name() << "\n";

        // Replace each input that consumes StructPack with decomposed tensors
        // Insert StructPack for each output that has type str

        // Inputs
        const auto& inputs = node->inputs();
        OutputVector new_inputs;
        bool at_least_one = false;
        for(size_t i = 0; i < inputs.size(); ++i) {
            auto input = inputs[i];
            if(ov::is_type<StructPack>(inputs[i].get_source_output().get_node_shared_ptr())) {
                // Don't really need to check StructPack output type
                // Because it should work for any structal type
                // TODO: Add rt_info to map input indices

                std::cerr << "[ 1 ]\n";
                auto input_inputs = get_inputs(inputs[i].get_source_output().get_node_shared_ptr());
                new_inputs.insert(new_inputs.end(), input_inputs.begin(), input_inputs.end());
                at_least_one = true;
            } else {
                std::cerr << "[ 2 ]\n";
                new_inputs.push_back(input.get_source_output());
            }
        }

        if(!at_least_one) {
                std::cerr << "[ 3 ]\n";
            return false;
        }

                        std::cerr << "[ 4 ]\n";


        auto new_node = node->clone_with_new_inputs(new_inputs);
        // new_node should have an extended set of outputs due to lowering semantics
        // inside the node; so the operation behind the node should expect lowered input set
        // and react accordingly by providing lowered outputs

        // Outputs
        const auto& outputs = node->outputs();
        if(outputs.size() == 1 && new_node->outputs().size() == 3) {
                            std::cerr << "[ 5 ]\n";

            // Suppose str tensor in the output
            auto output_shape = new_node->output(0).get_partial_shape();
            new_node = make_shared<StructPack>(
                new_node->outputs(),
                element::StructuralType::Str(), // parameter->get_rt_info()["structural_type"].as<StructuralTypeAttribute>().value
                output_shape);
        } else {
            std::cerr << "[ ERROR ] Multiple outputs unsupported\n";
            for(size_t i = 0; i < outputs.size(); ++i) {
            }
        }

                std::cerr << "[ 6 ]\n";

        replace_node(node, new_node);

        return false;
    };

    auto m = make_shared<ov::pass::pattern::Matcher>(node, "ov::frontend::tensorflow::pass::ThroughStrOpsProp");
    register_matcher(m, callback);
}


bool DecomposeStructResults::run_on_model(const std::shared_ptr<Model>& model) {
    // Search for Parameter with List[Tensor] types

    bool at_least_one_decomposed = false;

    ResultVector results =
        model->get_results();  // make a copy, leter results in the model are going to be modified

    for (size_t i = 0; i < results.size(); ++i) {
        auto result = results[i];
        if(is_type<StructPack>(result->get_input_node_ptr(0)))
        {
            auto inputs = result->get_input_node_ptr(0)->inputs();
            for (auto input : inputs) {
                model->add_results({make_shared<opset9::Result>(input.get_source_output())});
                // TODO: Keep tracking between original and new Results
                // FIXME: results order is changed
            }

            model->remove_result(result);
            at_least_one_decomposed = true;
        }
    }

    return at_least_one_decomposed;
}


}
}
}
}
