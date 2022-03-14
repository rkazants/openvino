// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mvn_fusion.hpp"

#include <memory>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

template <class T>
std::function<bool(ngraph::Output<ngraph::Node>)> value_is_equal_to(const std::vector<T>& ref_values) {
    return [ref_values](ngraph::Output<ngraph::Node> output) -> bool {
        auto node = output.get_node_shared_ptr();
        if (auto const_node = std::dynamic_pointer_cast<ngraph::op::Constant>(node)) {
            return const_node->template cast_vector<T>() == ref_values;
        }
        return false;
    };
}

ngraph::pass::MVNFusionWithoutConstants::MVNFusionWithoutConstants() {
    MATCHER_SCOPE(MVNFusionWithoutConstants);
    // Detect MVN decomposition pattern:
    // (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps)
    auto x = pattern::any_input();

    // (x - ReduceMean(x, axes))
    //     `------mean1-------'
    auto mean1_axes = pattern::wrap_type<opset6::Constant>();
    auto mean1 = pattern::wrap_type<opset6::ReduceMean>({x, mean1_axes});

    // (x - ReduceMean(x, axes))
    // `-sub1------------------'
    auto sub1 = pattern::wrap_type<opset6::Subtract>({x, mean1});

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                     `---mean2----------'
    auto mean2_axes = pattern::wrap_type<opset6::Constant>();
    auto mean2 = pattern::wrap_type<opset6::ReduceMean>({x, mean2_axes});

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                 `-sub2------------------'
    auto sub2 = pattern::wrap_type<opset6::Subtract>({x, mean2});

    const auto reuseSub1OrNot = std::make_shared<pattern::op::Or>(OutputVector{sub1, sub2});

    auto cast = pattern::wrap_type<opset6::Convert>({reuseSub1OrNot});
    const auto hasConvertOrNot = std::make_shared<pattern::op::Or>(OutputVector{cast, reuseSub1OrNot});

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                 `---------------------power--'
    auto const_2 = pattern::wrap_type<opset6::Constant>(value_is_equal_to<float>({2.0}));
    auto power = pattern::wrap_type<opset6::Power>({hasConvertOrNot, const_2});

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //     `---mean3--------------------------------'
    auto mean3_axes = pattern::wrap_type<opset6::Constant>();
    auto mean3 = pattern::wrap_type<opset6::ReduceMean>({power, mean3_axes});

    auto const_0_5 = pattern::wrap_type<ngraph::opset6::Constant>(value_is_equal_to<float>({0.5}));
    auto eps = pattern::wrap_type<opset6::Constant>();
    // ------------------- OUTSIDE_SQRT ----------------------

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    // `--Power--------------------------------------'
    auto power_sqrt_os = pattern::wrap_type<opset6::Power>({mean3, const_0_5});
    auto sqrt_os = pattern::wrap_type<opset6::Sqrt>({mean3});
    const auto powerOrSqrt_os = std::make_shared<pattern::op::Or>(OutputVector{power_sqrt_os, sqrt_os});

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps
    // `----------------------------------------------Add---'
    auto add_eps_os = pattern::wrap_type<opset6::Add>({powerOrSqrt_os, eps});

    // ------------------- INSIDE_SQRT ----------------------

    // (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps))
    // `-----------------------------------------------Add---'
    auto add_eps_is = pattern::wrap_type<opset6::Add>({mean3, eps});

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    // `--Power--------------------------------------'
    auto power_sqrt_is = pattern::wrap_type<opset6::Power>({add_eps_is, const_0_5});
    auto sqrt_is = pattern::wrap_type<opset6::Sqrt>({add_eps_is});
    const auto powerOrSqrt_is = std::make_shared<pattern::op::Or>(OutputVector{power_sqrt_is, sqrt_is});

    auto outsideOrInside = std::make_shared<pattern::op::Or>(OutputVector{add_eps_os, powerOrSqrt_is});

    // Final Divide
    auto const_neg_1 = pattern::wrap_type<opset6::Constant>(value_is_equal_to<float>({-1}));
    auto power_div = pattern::wrap_type<opset6::Power>({outsideOrInside, const_neg_1});
    auto div = pattern::wrap_type<opset6::Multiply>({sub1, power_div});

    auto div_alt = pattern::wrap_type<opset6::Divide>({sub1, outsideOrInside});
    const auto powerMulOrDiv = std::make_shared<pattern::op::Or>(OutputVector{div, div_alt});

    ngraph::matcher_pass_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto exp_input = pattern_to_output.at(x);

        auto const_eps_node =
            std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(eps).get_node_shared_ptr());
        float eps_value;
        if (!op::util::get_single_value(const_eps_node, eps_value)) {
            return false;
        }

        auto axes_1_node =
            std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(mean1_axes).get_node_shared_ptr());
        auto axes_3_node =
            std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(mean3_axes).get_node_shared_ptr());

        if (!axes_1_node || !axes_3_node) {
            return false;
        }

        auto axes_1_value = axes_1_node->cast_vector<int64_t>();
        auto axes_3_value = axes_3_node->cast_vector<int64_t>();

        if (axes_1_value != axes_3_value) {
            return false;
        }
        if (pattern_to_output.count(mean2_axes)) {
            auto axes_2_node = std::dynamic_pointer_cast<ngraph::opset6::Constant>(
                pattern_to_output.at(mean2_axes).get_node_shared_ptr());
            if (!axes_2_node) {
                return false;
            }
            auto axes_2_value = axes_2_node->cast_vector<int64_t>();
            if (axes_1_value != axes_2_value) {
                return false;
            }
        }

        ngraph::NodeVector nodes_to_copy_info({pattern_to_output.at(mean1).get_node_shared_ptr(),
                                               pattern_to_output.at(sub1).get_node_shared_ptr(),
                                               pattern_to_output.at(power).get_node_shared_ptr(),
                                               pattern_to_output.at(mean3).get_node_shared_ptr()});

        op::MVNEpsMode mode;
        if (pattern_to_output.count(add_eps_os)) {
            mode = op::MVNEpsMode::OUTSIDE_SQRT;
            nodes_to_copy_info.push_back(pattern_to_output.at(add_eps_os).get_node_shared_ptr());
            if (pattern_to_output.count(power_sqrt_os)) {
                nodes_to_copy_info.push_back(pattern_to_output.at(power_sqrt_os).get_node_shared_ptr());
            } else if (pattern_to_output.count(sqrt_os)) {
                nodes_to_copy_info.push_back(pattern_to_output.at(sqrt_os).get_node_shared_ptr());
            }
        } else if (pattern_to_output.count(powerOrSqrt_is)) {
            mode = op::MVNEpsMode::INSIDE_SQRT;
            nodes_to_copy_info.push_back(pattern_to_output.at(add_eps_is).get_node_shared_ptr());
            if (pattern_to_output.count(power_sqrt_is)) {
                nodes_to_copy_info.push_back(pattern_to_output.at(power_sqrt_is).get_node_shared_ptr());
            } else if (pattern_to_output.count(sqrt_is)) {
                nodes_to_copy_info.push_back(pattern_to_output.at(sqrt_is).get_node_shared_ptr());
            }
        } else {
            return false;
        }
        auto mvn = std::make_shared<ngraph::opset6::MVN>(exp_input, axes_1_node, true, eps_value, mode);

        if (pattern_to_output.count(mean2) && pattern_to_output.count(sub2)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(mean2).get_node_shared_ptr());
            nodes_to_copy_info.push_back(pattern_to_output.at(sub2).get_node_shared_ptr());
        }

        if (pattern_to_output.count(cast)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(cast).get_node_shared_ptr());
        }

        if (pattern_to_output.count(div_alt)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(div_alt).get_node_shared_ptr());
        } else if (pattern_to_output.count(power_div) && pattern_to_output.count(div)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(power_div).get_node_shared_ptr());
            nodes_to_copy_info.push_back(pattern_to_output.at(div).get_node_shared_ptr());
        }

        mvn->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(nodes_to_copy_info, mvn);
        ngraph::replace_node(m.get_match_root(), mvn);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(powerMulOrDiv, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ngraph::pass::MVNFusionWithConstantsInside::MVNFusionWithConstantsInside() {
    MATCHER_SCOPE(MVNFusionWithConstantsInside);
    // Detect MVN decomposition pattern:
    // (x - ReduceMean(x, axes)) * gamma / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps) + beta
    auto x = pattern::any_input();

    // (x - ReduceMean(x, axes))^2
    //     `------mean1-------'
    auto mean1_axes = pattern::wrap_type<opset6::Constant>();
    auto mean1 = pattern::wrap_type<opset6::ReduceMean>({x, mean1_axes});

    // (x - ReduceMean(x, axes))^2
    // `-squared_difference------'
    auto squared_difference = pattern::wrap_type<opset6::SquaredDifference>({x, mean1});

    // 1 / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    //         `---mean2--------------------------------'
    auto mean2_axes = pattern::wrap_type<opset6::Constant>();
    auto mean2 = pattern::wrap_type<opset6::ReduceMean>({squared_difference, mean2_axes});

    // 1 / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    //         `------------------------------------------add--'
    auto eps = pattern::wrap_type<opset6::Constant>();
    auto add_eps = pattern::wrap_type<opset6::Add>({mean2, eps});

    // 1 / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    // `-power-------------------------------------------------'
    auto const_0_5 = pattern::wrap_type<opset6::Constant>(value_is_equal_to<float>({-0.5}));
    auto power = pattern::wrap_type<opset6::Power>({add_eps, const_0_5});

    // gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    // `---mul1----------------------------------------------------'
    auto gamma = pattern::wrap_type<opset6::Constant>();
    auto mul1 = pattern::wrap_type<opset6::Multiply>({power, gamma});

    // x * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    // `---mul2--------------------------------------------------------'
    auto mul2 = pattern::wrap_type<opset6::Multiply>({x, mul1});

    // ReduceMean(x, axes) * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps) - beta
    // `-------------------mul3----------------------------------------------------------'
    auto mul3 = pattern::wrap_type<opset6::Multiply>({mul1, mean1});

    // beta - ReduceMean(x, axes) * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    // `---sub-----------------------------------------------------------------------------------'
    auto beta = pattern::wrap_type<opset6::Constant>();
    auto sub = pattern::wrap_type<opset6::Subtract>({beta, mul3});

    // Final Add
    // x * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps) +
    // beta - ReduceMean(x, axes) * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps) =
    // gamma * (x - ReduceMean(x, axes)) / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps) + beta
    auto add = pattern::wrap_type<opset6::Add>({mul2, sub});

    ngraph::matcher_pass_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(x);

        auto const_0_5_node =
            std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(const_0_5).get_node_shared_ptr());
        auto const_gamma_node =
            std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(gamma).get_node_shared_ptr());
        auto const_beta_node =
            std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(beta).get_node_shared_ptr());
        auto const_eps_node =
            std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(eps).get_node_shared_ptr());
        if (!const_0_5_node || !const_beta_node || !const_gamma_node || !const_eps_node) {
            return false;
        }

        float eps_value;
        bool valid_constant_values = op::util::has_constant_value<float>(const_0_5_node, -0.5) &&
                                     op::util::get_single_value(const_eps_node, eps_value);
        if (!valid_constant_values) {
            return false;
        }

        auto axes_1_node =
            std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(mean1_axes).get_node_shared_ptr());
        auto axes_2_node =
            std::dynamic_pointer_cast<ngraph::opset6::Constant>(pattern_to_output.at(mean2_axes).get_node_shared_ptr());
        if (!axes_1_node || !axes_2_node) {
            return false;
        }

        auto axes_1_value = axes_1_node->cast_vector<int64_t>();
        auto axes_2_value = axes_2_node->cast_vector<int64_t>();
        if (axes_1_value != axes_2_value) {
            return false;
        }

        auto mvn =
            std::make_shared<ngraph::opset6::MVN>(x_output, axes_1_node, true, eps_value, op::MVNEpsMode::INSIDE_SQRT);
        auto mul_gamma = std::make_shared<ngraph::opset6::Multiply>(mvn, const_gamma_node);
        auto add_beta = std::make_shared<ngraph::opset6::Add>(mul_gamma, const_beta_node);

        ngraph::copy_runtime_info({pattern_to_output.at(mean1).get_node_shared_ptr(),
                                   pattern_to_output.at(squared_difference).get_node_shared_ptr(),
                                   pattern_to_output.at(add_eps).get_node_shared_ptr(),
                                   pattern_to_output.at(power).get_node_shared_ptr(),
                                   pattern_to_output.at(mul1).get_node_shared_ptr(),
                                   pattern_to_output.at(mul2).get_node_shared_ptr(),
                                   pattern_to_output.at(mul3).get_node_shared_ptr(),
                                   pattern_to_output.at(sub).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr()},
                                  {mvn, const_gamma_node, mul_gamma, const_beta_node, add_beta});
        add_beta->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::replace_node(m.get_match_root(), add_beta);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
