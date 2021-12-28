// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void regclass_frontend_FrontEndManager(py::module m);
void regclass_frontend_NotImplementedFailureFrontEnd(py::module m);
void regclass_frontend_InitializationFailureFrontEnd(py::module m);
void regclass_frontend_OpConversionFailureFrontEnd(py::module m);
void regclass_frontend_OpValidationFailureFrontEnd(py::module m);
void regclass_frontend_GeneralFailureFrontEnd(py::module m);
