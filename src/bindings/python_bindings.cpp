#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/VectorStore.h"

namespace py = pybind11;
using namespace vectordb;

PYBIND11_MODULE(vectordb_py, m) {
  m.doc() = "GPU-Accelerated Vector Database Engine — Python bindings";

  // ── MetricType enum ────────────────────────────────────────────────────────
  py::enum_<MetricType>(m, "MetricType")
    .value("L2",           MetricType::L2)
    .value("Cosine",       MetricType::Cosine)
    .value("InnerProduct", MetricType::InnerProduct)
    .export_values();

  // ── VectorStoreConfig ──────────────────────────────────────────────────────
  py::class_<VectorStoreConfig>(m, "VectorStoreConfig")
    .def(py::init<size_t, MetricType, size_t>(),
         py::arg("dim"),
         py::arg("metric")   = MetricType::L2,
         py::arg("capacity") = 1'000'000)
    .def_readwrite("dim",        &VectorStoreConfig::dim)
    .def_readwrite("metric",     &VectorStoreConfig::metric)
    .def_readwrite("capacity",   &VectorStoreConfig::capacity)
    .def_readwrite("gpu_enabled",&VectorStoreConfig::gpu_enabled)
    .def_readwrite("gpu_device", &VectorStoreConfig::gpu_device);

  // ── VectorStore ────────────────────────────────────────────────────────────
  py::class_<VectorStore>(m, "VectorStore")
    .def(py::init<const VectorStoreConfig&>(), py::arg("config"))
    .def("insert",       &VectorStore::insert,       py::arg("vec"))
    .def("insert_batch", &VectorStore::insert_batch, py::arg("vecs"))
    .def("search",       &VectorStore::search,
                         py::arg("query"), py::arg("k"))
    .def("dim",          &VectorStore::dim)
    .def("size",         &VectorStore::size)
    .def("capacity",     &VectorStore::capacity)
    .def("gpu_enabled",  &VectorStore::gpu_enabled)
    .def("info",         &VectorStore::info)
    .def("__repr__",     &VectorStore::info);
}