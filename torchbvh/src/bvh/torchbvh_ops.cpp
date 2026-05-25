#include "bvh/bvh.h"

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <memory>
#include <stdexcept>
#include <vector>

namespace thbvh {
namespace {

void check_cuda_contiguous(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous.");
}

void check_shape2(const torch::Tensor& t, int64_t d1, const char* name) {
  TORCH_CHECK(t.dim() == 2, name, " must be rank-2.");
  TORCH_CHECK(t.size(1) == d1, name, " second dim must be ", d1, ".");
}

void check_shape1(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.dim() == 1, name, " must be rank-1.");
}

Eigen::MatrixXd to_eigen_vertices(const torch::Tensor& vertices_cpu_f64) {
  Eigen::MatrixXd out(vertices_cpu_f64.size(0), 3);
  const double* ptr = vertices_cpu_f64.data_ptr<double>();
  for (int64_t i = 0; i < vertices_cpu_f64.size(0); ++i) {
    out(i, 0) = ptr[i * 3 + 0];
    out(i, 1) = ptr[i * 3 + 1];
    out(i, 2) = ptr[i * 3 + 2];
  }
  return out;
}

Eigen::MatrixXi to_eigen_faces(const torch::Tensor& faces_cpu_i32) {
  Eigen::MatrixXi out(faces_cpu_i32.size(0), 3);
  const int32_t* ptr = faces_cpu_i32.data_ptr<int32_t>();
  for (int64_t i = 0; i < faces_cpu_i32.size(0); ++i) {
    out(i, 0) = ptr[i * 3 + 0];
    out(i, 1) = ptr[i * 3 + 1];
    out(i, 2) = ptr[i * 3 + 2];
  }
  return out;
}

}  // namespace

std::vector<torch::Tensor> build_bvh_tensors(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    int64_t max_leaf_size) {
  TORCH_CHECK(vertices.dim() == 2 && vertices.size(1) == 3, "vertices must be [V,3].");
  TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3, "faces must be [F,3].");
  TORCH_CHECK(max_leaf_size > 0, "max_leaf_size must be > 0.");

  auto vertices_cpu =
      vertices.to(torch::kCPU, torch::kFloat64, false, true).contiguous();
  auto faces_cpu = faces.to(torch::kCPU, torch::kInt32, false, true).contiguous();
  const Eigen::MatrixXd eigen_vertices = to_eigen_vertices(vertices_cpu);
  const Eigen::MatrixXi eigen_faces = to_eigen_faces(faces_cpu);

  torch::Device upload_device = torch::Device(torch::kCUDA, 0);
  if (vertices.is_cuda()) {
    upload_device = vertices.device();
  }
  THBVH bvh(eigen_vertices, eigen_faces, static_cast<int>(max_leaf_size));
  bvh.upload(upload_device);

  return {
      bvh.node_lower_tensor(),
      bvh.node_upper_tensor(),
      bvh.primitive_indices_tensor(),
      bvh.faces_tensor(),
      bvh.vertices_tensor()};
}

torch::Tensor point_mesh_query(
    const torch::Tensor& points,
    const torch::Tensor& node_lower,
    const torch::Tensor& node_upper,
    const torch::Tensor& primitive_indices,
    const torch::Tensor& faces,
    const torch::Tensor& vertices);

torch::Tensor ray_mesh_query(
    const torch::Tensor& ray_origins,
    const torch::Tensor& ray_dirs,
    const torch::Tensor& node_lower,
    const torch::Tensor& node_upper,
    const torch::Tensor& primitive_indices,
    const torch::Tensor& faces,
    const torch::Tensor& vertices);

struct BVHBinding {
  explicit BVHBinding(std::vector<torch::Tensor> tensors)
      : node_lower(std::move(tensors[0])),
        node_upper(std::move(tensors[1])),
        primitive_indices(std::move(tensors[2])),
        faces(std::move(tensors[3])),
        vertices(std::move(tensors[4])) {}

  torch::Tensor query(const torch::Tensor& points) const {
    return point_mesh_query(
        points,
        node_lower,
        node_upper,
        primitive_indices,
        faces,
        vertices);
  }

  torch::Tensor ray_query(const torch::Tensor& ray_origins, const torch::Tensor& ray_dirs) const {
    return ray_mesh_query(
        ray_origins,
        ray_dirs,
        node_lower,
        node_upper,
        primitive_indices,
        faces,
        vertices);
  }

  torch::Tensor node_lower;
  torch::Tensor node_upper;
  torch::Tensor primitive_indices;
  torch::Tensor faces;
  torch::Tensor vertices;
};

std::shared_ptr<BVHBinding> build_bvh_binding(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    int64_t max_leaf_size) {
  return std::make_shared<BVHBinding>(build_bvh_tensors(vertices, faces, max_leaf_size));
}

torch::Tensor point_mesh_query(
    const torch::Tensor& points,
    const torch::Tensor& node_lower,
    const torch::Tensor& node_upper,
    const torch::Tensor& primitive_indices,
    const torch::Tensor& faces,
    const torch::Tensor& vertices) {
  check_cuda_contiguous(points, "points");
  check_cuda_contiguous(node_lower, "node_lower");
  check_cuda_contiguous(node_upper, "node_upper");
  check_cuda_contiguous(primitive_indices, "primitive_indices");
  check_cuda_contiguous(faces, "faces");
  check_cuda_contiguous(vertices, "vertices");

  TORCH_CHECK(points.scalar_type() == torch::kFloat32, "points must be float32.");
  TORCH_CHECK(node_lower.scalar_type() == torch::kFloat32, "node_lower must be float32.");
  TORCH_CHECK(node_upper.scalar_type() == torch::kFloat32, "node_upper must be float32.");
  TORCH_CHECK(primitive_indices.scalar_type() == torch::kInt32, "primitive_indices must be int32.");
  TORCH_CHECK(faces.scalar_type() == torch::kInt32, "faces must be int32.");
  TORCH_CHECK(vertices.scalar_type() == torch::kFloat32, "vertices must be float32.");

  check_shape2(points, 3, "points");
  check_shape2(node_lower, 4, "node_lower");
  check_shape2(node_upper, 4, "node_upper");
  check_shape2(faces, 3, "faces");
  check_shape2(vertices, 3, "vertices");
  check_shape1(primitive_indices, "primitive_indices");

  const int64_t node_count = node_lower.size(0);
  TORCH_CHECK(node_upper.size(0) == node_count, "node_upper rows mismatch.");

  return point_mesh_query_cuda(
      points,
      node_lower,
      node_upper,
      primitive_indices,
      faces,
      vertices);
}

torch::Tensor ray_mesh_query(
    const torch::Tensor& ray_origins,
    const torch::Tensor& ray_dirs,
    const torch::Tensor& node_lower,
    const torch::Tensor& node_upper,
    const torch::Tensor& primitive_indices,
    const torch::Tensor& faces,
    const torch::Tensor& vertices) {
  check_cuda_contiguous(ray_origins, "ray_origins");
  check_cuda_contiguous(ray_dirs, "ray_dirs");
  check_cuda_contiguous(node_lower, "node_lower");
  check_cuda_contiguous(node_upper, "node_upper");
  check_cuda_contiguous(primitive_indices, "primitive_indices");
  check_cuda_contiguous(faces, "faces");
  check_cuda_contiguous(vertices, "vertices");

  TORCH_CHECK(ray_origins.scalar_type() == torch::kFloat32, "ray_origins must be float32.");
  TORCH_CHECK(ray_dirs.scalar_type() == torch::kFloat32, "ray_dirs must be float32.");
  TORCH_CHECK(node_lower.scalar_type() == torch::kFloat32, "node_lower must be float32.");
  TORCH_CHECK(node_upper.scalar_type() == torch::kFloat32, "node_upper must be float32.");
  TORCH_CHECK(primitive_indices.scalar_type() == torch::kInt32, "primitive_indices must be int32.");
  TORCH_CHECK(faces.scalar_type() == torch::kInt32, "faces must be int32.");
  TORCH_CHECK(vertices.scalar_type() == torch::kFloat32, "vertices must be float32.");

  check_shape2(ray_origins, 3, "ray_origins");
  check_shape2(ray_dirs, 3, "ray_dirs");
  check_shape2(node_lower, 4, "node_lower");
  check_shape2(node_upper, 4, "node_upper");
  check_shape2(faces, 3, "faces");
  check_shape2(vertices, 3, "vertices");
  check_shape1(primitive_indices, "primitive_indices");
  TORCH_CHECK(ray_origins.size(0) == ray_dirs.size(0), "ray_origins/ray_dirs rows mismatch.");

  const int64_t node_count = node_lower.size(0);
  TORCH_CHECK(node_upper.size(0) == node_count, "node_upper rows mismatch.");

  return ray_mesh_query_cuda(
      ray_origins,
      ray_dirs,
      node_lower,
      node_upper,
      primitive_indices,
      faces,
      vertices);
}

}  // namespace thbvh

TORCH_LIBRARY(torchbvh, m) {
  m.def("build_bvh(Tensor vertices, Tensor faces, int max_leaf_size=8) -> Tensor[]");
  m.def(
      "point_mesh_query(Tensor points, Tensor node_lower, Tensor node_upper, "
      "Tensor primitive_indices, Tensor faces, Tensor vertices) -> Tensor");
  m.def(
      "ray_mesh_query(Tensor ray_origins, Tensor ray_dirs, Tensor node_lower, Tensor node_upper, "
      "Tensor primitive_indices, Tensor faces, Tensor vertices) -> Tensor");
}

TORCH_LIBRARY_IMPL(torchbvh, CPU, m) {
  m.impl("build_bvh", &thbvh::build_bvh_tensors);
}

TORCH_LIBRARY_IMPL(torchbvh, CUDA, m) {
  m.impl("build_bvh", &thbvh::build_bvh_tensors);
  m.impl("point_mesh_query", &thbvh::point_mesh_query);
  m.impl("ray_mesh_query", &thbvh::ray_mesh_query);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  py::class_<thbvh::BVHBinding, std::shared_ptr<thbvh::BVHBinding>>(m, "BVH")
      .def("query", &thbvh::BVHBinding::query)
      .def("ray_query", &thbvh::BVHBinding::ray_query)
      .def_property_readonly(
          "node_lower",
          [](const thbvh::BVHBinding& self) { return self.node_lower; })
      .def_property_readonly(
          "node_upper",
          [](const thbvh::BVHBinding& self) { return self.node_upper; })
      .def_property_readonly(
          "primitive_indices",
          [](const thbvh::BVHBinding& self) { return self.primitive_indices; })
      .def_property_readonly("faces", [](const thbvh::BVHBinding& self) { return self.faces; })
      .def_property_readonly("vertices", [](const thbvh::BVHBinding& self) { return self.vertices; });

  m.def("build_bvh", &thbvh::build_bvh_binding, py::arg("vertices"), py::arg("faces"), py::arg("max_leaf_size") = 8);
}
