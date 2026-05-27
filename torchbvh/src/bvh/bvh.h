#pragma once

#include <torch/torch.h>

#include "bvh/bvh_host.h"

namespace thbvh {

class THBVH {
 public:
  THBVH(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces, int max_leaf_size = 8);
  ~THBVH();

  void rebuild(int max_leaf_size = 8);
  void upload(torch::Device device = torch::kCUDA);

  bool uploaded() const { return m_uploaded; }
  int max_leaf_size() const { return m_max_leaf_size; }

  const torch::Tensor& node_lower_tensor() const { return m_node_lower_tensor; }
  const torch::Tensor& node_upper_tensor() const { return m_node_upper_tensor; }
  const torch::Tensor& primitive_indices_tensor() const { return m_primitive_indices_tensor; }
  const torch::Tensor& faces_tensor() const { return m_faces_tensor; }
  const torch::Tensor& vertices_tensor() const { return m_vertices_tensor; }

 private:
  void build_cpu_flattened();

  Eigen::MatrixXd m_vertices;
  Eigen::MatrixXi m_faces;

  BVHNode4* m_root = nullptr;
  int m_max_leaf_size = 8;
  bool m_uploaded = false;

  std::vector<FlattenedNode4> m_flat_nodes;
  std::vector<int> m_primitive_indices;

  torch::Tensor m_node_lower_tensor;  // [N,4]: min.xyz + left(bits)
  torch::Tensor m_node_upper_tensor;  // [N,4]: max.xyz + right(bits)
  torch::Tensor m_primitive_indices_tensor;
  torch::Tensor m_faces_tensor;
  torch::Tensor m_vertices_tensor;
};

torch::Tensor point_mesh_query_cuda(
    const torch::Tensor& points,
    const torch::Tensor& node_lower,
    const torch::Tensor& node_upper,
    const torch::Tensor& primitive_indices,
    const torch::Tensor& faces,
    const torch::Tensor& vertices);

torch::Tensor ray_mesh_query_cuda(
    const torch::Tensor& ray_origins,
    const torch::Tensor& ray_dirs,
    const torch::Tensor& node_lower,
    const torch::Tensor& node_upper,
    const torch::Tensor& primitive_indices,
    const torch::Tensor& faces,
    const torch::Tensor& vertices);

std::vector<torch::Tensor> box_overlap_query_cuda(
    const torch::Tensor& box_lower,
    const torch::Tensor& box_upper,
    const torch::Tensor& node_lower,
    const torch::Tensor& node_upper,
    const torch::Tensor& primitive_indices,
    const torch::Tensor& faces,
    const torch::Tensor& vertices);

}  // namespace thbvh