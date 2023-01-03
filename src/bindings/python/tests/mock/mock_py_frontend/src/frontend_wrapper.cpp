// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_py_frontend/frontend_wrappers.hpp"

#ifdef ENABLE_OV_TF_FRONTEND
FrontEndWrapperTensorflow::FrontEndWrapperTensorflow() = default;

void FrontEndWrapperTensorflow::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    FrontEnd::add_extension(extension);
}

bool FrontEndWrapperTensorflow::check_conversion_extension_registered(const std::string& name) {
    bool is_found = false;
    for (const auto& conversion : m_conversion_extensions) {
        if (conversion->get_op_type() == name) {
            is_found = true;
            break;
        }
    }
    return is_found;
}
#endif

#ifdef ENABLE_OV_PADDLE_FRONTEND
FrontEndWrapperPaddle::FrontEndWrapperPaddle() = default;

void FrontEndWrapperPaddle::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    FrontEnd::add_extension(extension);
}

bool FrontEndWrapperPaddle::check_conversion_extension_registered(const std::string& name) {
    return m_op_translators.find(name) != m_op_translators.end();
}
#endif
