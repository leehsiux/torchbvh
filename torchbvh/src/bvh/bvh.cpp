#include "bvh/bvh.h"

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
  std::vector<float> node_min(static_cast<size_t>(node_count * 3));
  std::vector<float> node_max(static_cast<size_t>(node_count * 3));
  std::vector<int32_t> node_left(static_cast<size_t>(node_count));
  std::vector<int32_t> node_right(static_cast<size_t>(node_count));
  std::vector<int32_t> node_child_count(static_cast<size_t>(node_count));

  for (int64_t i = 0; i < node_count; ++i) {
    const FlattenedNode4& node = m_flat_nodes[static_cast<size_t>(i)];
    node_min[static_cast<size_t>(i * 3 + 0)] = static_cast<float>(node.m_min.x());
    node_min[static_cast<size_t>(i * 3 + 1)] = static_cast<float>(node.m_min.y());
    node_min[static_cast<size_t>(i * 3 + 2)] = static_cast<float>(node.m_min.z());
    node_max[static_cast<size_t>(i * 3 + 0)] = static_cast<float>(node.m_max.x());
    node_max[static_cast<size_t>(i * 3 + 1)] = static_cast<float>(node.m_max.y());
    node_max[static_cast<size_t>(i * 3 + 2)] = static_cast<float>(node.m_max.z());
    node_left[static_cast<size_t>(i)] = node.m_left;
    node_right[static_cast<size_t>(i)] = node.m_right;
    node_child_count[static_cast<size_t>(i)] = node.m_child_count;
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

  m_node_min_tensor =
      torch::from_blob(node_min.data(), {node_count, 3}, f32).clone().contiguous();
  m_node_max_tensor =
      torch::from_blob(node_max.data(), {node_count, 3}, f32).clone().contiguous();
  m_node_left_tensor = torch::from_blob(node_left.data(), {node_count}, i32).clone().contiguous();
  m_node_right_tensor =
      torch::from_blob(node_right.data(), {node_count}, i32).clone().contiguous();
  m_node_child_count_tensor =
      torch::from_blob(node_child_count.data(), {node_count}, i32).clone().contiguous();

  m_primitive_indices_tensor = torch::from_blob(
                                   primitive_indices.data(),
                                   {static_cast<int64_t>(primitive_indices.size())},
                                   i32)
                                   .clone()
                                   .contiguous();
  m_faces_tensor = torch::from_blob(faces_data.data(), {face_count, 3}, i32).clone().contiguous();
  m_vertices_tensor =
      torch::from_blob(vertices_data.data(), {vertex_count, 3}, f32).clone().contiguous();

  m_node_min_tensor = to_device(m_node_min_tensor, device);
  m_node_max_tensor = to_device(m_node_max_tensor, device);
  m_node_left_tensor = to_device(m_node_left_tensor, device);
  m_node_right_tensor = to_device(m_node_right_tensor, device);
  m_node_child_count_tensor = to_device(m_node_child_count_tensor, device);
  m_primitive_indices_tensor = to_device(m_primitive_indices_tensor, device);
  m_faces_tensor = to_device(m_faces_tensor, device);
  m_vertices_tensor = to_device(m_vertices_tensor, device);

  m_uploaded = true;
}

}  // namespace thbvh
