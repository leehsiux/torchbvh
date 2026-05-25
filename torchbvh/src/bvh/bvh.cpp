#include "bvh/bvh.h"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace thbvh {

namespace {
torch::Tensor to_device(const torch::Tensor& cpu_tensor, const torch::Device& device) {
  if (cpu_tensor.device() == device) {
    return cpu_tensor;
  }
  return cpu_tensor.to(device, cpu_tensor.scalar_type(), false, true);
}

float bitcast_i32_to_f32(int32_t v) {
  float out = 0.0f;
  static_assert(sizeof(out) == sizeof(v), "bitcast size mismatch");
  std::memcpy(&out, &v, sizeof(out));
  return out;
}
}  // namespace

THBVH::THBVH(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces, int max_leaf_size)
    : m_vertices(vertices), m_faces(faces), m_max_leaf_size(std::max(1, max_leaf_size)) {
  build_cpu_flattened();
}

THBVH::~THBVH() {
  delete_bvh4(m_root);
  m_root = nullptr;
}

void THBVH::rebuild(int max_leaf_size) {
  m_max_leaf_size = std::max(1, max_leaf_size);
  build_cpu_flattened();
  m_uploaded = false;
}

void THBVH::build_cpu_flattened() {
  delete_bvh4(m_root);
  m_root = build_bvh4(m_vertices, m_faces, m_max_leaf_size);
  if (m_root == nullptr) {
    throw std::runtime_error("Cannot build BVH from empty mesh.");
  }
  flatten_bvh4(m_root, m_flat_nodes, m_primitive_indices);
}

void THBVH::upload(torch::Device device) {
  if (!device.is_cuda()) {
    throw std::invalid_argument("THBVH::upload expects a CUDA device.");
  }

  if (m_flat_nodes.empty()) {
    throw std::runtime_error("BVH is empty; rebuild before upload.");
  }

  const int64_t node_count = static_cast<int64_t>(m_flat_nodes.size());
  std::vector<float> node_lower(static_cast<size_t>(node_count * 4));
  std::vector<float> node_upper(static_cast<size_t>(node_count * 4));

  for (int64_t i = 0; i < node_count; ++i) {
    const FlattenedNode4& node = m_flat_nodes[static_cast<size_t>(i)];
    node_lower[static_cast<size_t>(i * 4 + 0)] = static_cast<float>(node.m_min.x());
    node_lower[static_cast<size_t>(i * 4 + 1)] = static_cast<float>(node.m_min.y());
    node_lower[static_cast<size_t>(i * 4 + 2)] = static_cast<float>(node.m_min.z());
    node_lower[static_cast<size_t>(i * 4 + 3)] = bitcast_i32_to_f32(node.m_left);
    node_upper[static_cast<size_t>(i * 4 + 0)] = static_cast<float>(node.m_max.x());
    node_upper[static_cast<size_t>(i * 4 + 1)] = static_cast<float>(node.m_max.y());
    node_upper[static_cast<size_t>(i * 4 + 2)] = static_cast<float>(node.m_max.z());
    node_upper[static_cast<size_t>(i * 4 + 3)] = bitcast_i32_to_f32(node.m_right);
  }

  std::vector<int32_t> primitive_indices(m_primitive_indices.size());
  for (size_t i = 0; i < m_primitive_indices.size(); ++i) {
    primitive_indices[i] = static_cast<int32_t>(m_primitive_indices[i]);
  }

  const int64_t face_count = m_faces.rows();
  std::vector<int32_t> faces_data(static_cast<size_t>(face_count * 3));
  for (int64_t i = 0; i < face_count; ++i) {
    faces_data[static_cast<size_t>(i * 3 + 0)] = static_cast<int32_t>(m_faces(i, 0));
    faces_data[static_cast<size_t>(i * 3 + 1)] = static_cast<int32_t>(m_faces(i, 1));
    faces_data[static_cast<size_t>(i * 3 + 2)] = static_cast<int32_t>(m_faces(i, 2));
  }

  const int64_t vertex_count = m_vertices.rows();
  std::vector<float> vertices_data(static_cast<size_t>(vertex_count * 3));
  for (int64_t i = 0; i < vertex_count; ++i) {
    vertices_data[static_cast<size_t>(i * 3 + 0)] = static_cast<float>(m_vertices(i, 0));
    vertices_data[static_cast<size_t>(i * 3 + 1)] = static_cast<float>(m_vertices(i, 1));
    vertices_data[static_cast<size_t>(i * 3 + 2)] = static_cast<float>(m_vertices(i, 2));
  }

  auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

  m_node_lower_tensor =
      torch::from_blob(node_lower.data(), {node_count, 4}, f32).clone().contiguous();
  m_node_upper_tensor =
      torch::from_blob(node_upper.data(), {node_count, 4}, f32).clone().contiguous();

  m_primitive_indices_tensor = torch::from_blob(
                                   primitive_indices.data(),
                                   {static_cast<int64_t>(primitive_indices.size())},
                                   i32)
                                   .clone()
                                   .contiguous();
  m_faces_tensor = torch::from_blob(faces_data.data(), {face_count, 3}, i32).clone().contiguous();
  m_vertices_tensor =
      torch::from_blob(vertices_data.data(), {vertex_count, 3}, f32).clone().contiguous();

  m_node_lower_tensor = to_device(m_node_lower_tensor, device);
  m_node_upper_tensor = to_device(m_node_upper_tensor, device);
  m_primitive_indices_tensor = to_device(m_primitive_indices_tensor, device);
  m_faces_tensor = to_device(m_faces_tensor, device);
  m_vertices_tensor = to_device(m_vertices_tensor, device);

  m_uploaded = true;
}

}  // namespace thbvh
