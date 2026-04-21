// SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
//
// SPDX-License-Identifier: Apache-2.0

#include "data_read.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(hpcanalysis_cpp, m) {
  m.doc() = "Pybind11 bindings for the C++ implementation of hpcanalysis' Data "
            "Read API";

  py::class_<DataRead>(m, "DataRead")
      .def(py::init<const std::string &>(), py::arg("dir_path"))
      .def("read_metric_descriptions", &DataRead::read_metric_descriptions)
      .def("read_profile_descriptions", &DataRead::read_profile_descriptions)
      .def("read_profile_slices", &DataRead::read_profile_slices,
           py::arg("profile_indices"), py::arg("cct_indices"),
           py::arg("metric_indices"), py::arg("thread_count"))
      .def("read_trace_slices", &DataRead::read_trace_slices,
           py::arg("profile_indices"), py::arg("thread_count"));
}