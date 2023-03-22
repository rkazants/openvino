// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdlib.h>

#include <fstream>
#include <string>

#include "graph_iterator_meta.hpp"
#include "openvino/core/type/element_type.hpp"
#include "tensor_bundle.pb.h"
#include "trackable_object_graph.pb.h"

#ifdef ENABLE_SNAPPY_COMPRESSION
#    include "snappy.h"
#endif

namespace ov {
namespace frontend {
namespace tensorflow {

bool GraphIteratorMeta::is_valid_signature(const ::tensorflow::SignatureDef& signature) const {
    const std::map<::tensorflow::DataType, ov::element::Type> types{
        {::tensorflow::DataType::DT_BOOL, ov::element::boolean},
        {::tensorflow::DataType::DT_INT16, ov::element::i16},
        {::tensorflow::DataType::DT_INT32, ov::element::i32},
        {::tensorflow::DataType::DT_INT64, ov::element::i64},
        {::tensorflow::DataType::DT_HALF, ov::element::f16},
        {::tensorflow::DataType::DT_FLOAT, ov::element::f32},
        {::tensorflow::DataType::DT_DOUBLE, ov::element::f64},
        {::tensorflow::DataType::DT_UINT8, ov::element::u8},
        {::tensorflow::DataType::DT_INT8, ov::element::i8},
        {::tensorflow::DataType::DT_BFLOAT16, ov::element::bf16},
        {::tensorflow::DataType::DT_STRING, ov::element::undefined}};

    for (const auto& it : signature.inputs()) {
        if (it.second.name().empty() || types.find(it.second.dtype()) == types.end())
            return false;
    }
    for (const auto& it : signature.outputs()) {
        if (it.second.name().empty() || types.find(it.second.dtype()) == types.end())
            return false;
    }
    return true;
}

bool GraphIteratorMeta::is_supported(const std::string& path) {
    std::ifstream mg_stream(path, std::ios::in | std::ifstream::binary);
    auto metagraph_def = std::make_shared<::tensorflow::MetaGraphDef>();
    return mg_stream && mg_stream.is_open() && metagraph_def->ParsePartialFromIstream(&mg_stream);
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
bool GraphIteratorMeta::is_supported(const std::wstring& path) {
    std::ifstream mg_stream(path, std::ios::in | std::ifstream::binary);
    auto metagraph_def = std::make_shared<::tensorflow::MetaGraphDef>();
    return mg_stream && mg_stream.is_open() && metagraph_def->ParsePartialFromIstream(&mg_stream);
}
#endif

template <>
std::basic_string<char> get_variables_index_name<char>(const std::string name) {
    return name + ".index";
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
std::basic_string<wchar_t> get_variables_index_name<wchar_t>(const std::wstring name) {
    return name + L".index";
}
#endif

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
