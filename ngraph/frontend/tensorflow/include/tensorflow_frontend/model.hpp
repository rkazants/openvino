// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// TODO: include it by just frontend_manager.hpp without path
#include <frontend_manager/frontend.hpp>
#include <frontend_manager/place.hpp>
#include <tensorflow_frontend/utility.hpp>

namespace tensorflow {
class GraphDef;
class NodeDef;
namespace ngraph_bridge {
class GraphIteratorProto;
}
}  // namespace tensorflow

namespace ngraph {
namespace frontend {

class OpPlaceTF;
class TensorPlaceTF;

/*
class TF_API InputModelTensorflow : public InputModel {
public:
    // TODO: move these members to private section
    std::shared_ptr<::tensorflow::ngraph_bridge::GraphIteratorProto> graph_impl;
    std::shared_ptr<::tensorflow::GraphDef> graph_def;
    std::string path;
    std::vector<ngraph::PartialShape> input_shapes;
    // TODO: map from PlaceTensorflow, not from name string
    std::map<std::string, ngraph::PartialShape> partialShapes;

    std::vector<std::shared_ptr<ngraph::frontend::OpPlaceTF>> get_op_places() const;

public:
    InputModelTensorflow(const std::string& _path);
    InputModelTensorflow(const std::vector<std::istream*>& streams);
    InputModelTensorflow(std::shared_ptr<::tensorflow::GraphDef> _graph_def,
                         std::vector<ngraph::PartialShape> _input_shapes = {});
    InputModelTensorflow(const std::vector<std::shared_ptr<::tensorflow::NodeDef>>& _nodes_def,
                         std::vector<ngraph::PartialShape> _input_shapes = {});

    std::vector<Place::Ptr> get_inputs() const override;
    std::vector<Place::Ptr> get_outputs() const override;

    Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const override {
        // TODO: implement
        return nullptr;
    }
    void override_all_outputs(const std::vector<Place::Ptr>& outputs) override {
        // TODO: implement
    }
    void override_all_inputs(const std::vector<Place::Ptr>& inputs) override {
        // TODO: implement
    }
    void extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) override {
        // TODO: implement
    }
    virtual void set_partial_shape(Place::Ptr place, const ngraph::PartialShape& pshape) override;
    virtual ngraph::PartialShape get_partial_shape(Place::Ptr place) const override;
    void set_element_type(Place::Ptr place, const ngraph::element::Type&) override{
        // TODO: implement
    };
    void set_tensor_value(Place::Ptr place, const void* value) override{
        // TODO: implement
    };

private:
    std::map<std::string, std::shared_ptr<OpPlaceTF>> m_ops;
    std::vector<std::shared_ptr<OpPlaceTF>> m_ops_topology_sorted;
    std::vector<Place::Ptr> m_inputs;
    std::vector<Place::Ptr> m_outputs;
    // traverse graph to find output ops
    void initial_traverse_graph();
    // traverse graph from outputs to inputs to get nodes remaining in graph
    std::vector<std::shared_ptr<OpPlaceTF>> determine_cut_nodes() const;
};
*/

class TF_API InputModelTF : public InputModel {
    friend class FrontEndTF;
    class InputModelTFImpl;
    std::shared_ptr<InputModelTFImpl> _impl;

    std::map<std::string, Output<Node>> get_tensor_values() const;

public:
    // TODO: move to private once Translation will be a part of FrontEndTF component
    std::map<std::string, std::shared_ptr<TensorPlaceTF>> get_var_places() const;
    std::vector<std::shared_ptr<OpPlaceTF>> get_op_places() const;

    explicit InputModelTF(const std::string& path);
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    explicit InputModelTF(const std::wstring& path);
#endif
    explicit InputModelTF(const std::vector<std::istream*>& streams);
    std::vector<Place::Ptr> get_inputs() const override;
    std::vector<Place::Ptr> get_outputs() const override;
    Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const override;
    void override_all_outputs(const std::vector<Place::Ptr>& outputs) override;
    void override_all_inputs(const std::vector<Place::Ptr>& inputs) override;
    void extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) override;
    void set_partial_shape(Place::Ptr place, const ngraph::PartialShape&) override;
    ngraph::PartialShape get_partial_shape(Place::Ptr place) const override;
    void set_element_type(Place::Ptr place, const ngraph::element::Type&) override;
    void set_tensor_value(Place::Ptr place, const void* value) override;
};

}  // namespace frontend
}  // namespace ngraph
