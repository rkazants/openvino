// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <tensorflow_frontend/decoder.hpp>
#include <tensorflow_frontend/graph_iterator.hpp>

#include "decoder_proto.hpp"
#include "graph.pb.h"
#include "node_def.pb.h"

namespace ngraph {
namespace frontend {
namespace tf {
class GraphIteratorProto : public GraphIterator {
    std::vector<const ::tensorflow::NodeDef*> m_nodes;
    size_t node_index = 0;
    std::shared_ptr<::tensorflow::GraphDef> m_graph_def;

public:
    template <typename T>
    GraphIteratorProto(const std::basic_string<T>& path) : m_graph_def(std::make_shared<::tensorflow::GraphDef>()) {
        std::ifstream pb_stream(path, std::ios::in | std::ifstream::binary);

        FRONT_END_GENERAL_CHECK(pb_stream && pb_stream.is_open(), "Model file does not exist");
        FRONT_END_GENERAL_CHECK(m_graph_def->ParseFromIstream(&pb_stream), "Model cannot be parsed");

        m_nodes.resize(m_graph_def->node_size());
        for (size_t i = 0; i < m_nodes.size(); ++i)
            m_nodes[i] = &m_graph_def->node(i);
    }

    /// Set iterator to the start position
    virtual void reset() override {
        node_index = 0;
    }

    virtual size_t size() const override {
        return m_nodes.size();
    }

    /// Moves to the next node in the graph
    virtual void next() override {
        node_index++;
    }

    virtual bool is_end() const override {
        return node_index >= m_nodes.size();
    }

    /// Return NodeContext for the current node that iterator points to
    virtual std::shared_ptr<DecoderBase> get_decoder() const override {
        return std::make_shared<DecoderTFProto>(m_nodes[node_index]);
    }
};
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
