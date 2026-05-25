#pragma once

#if __has_include(<Eigen/Core>)
#include <Eigen/Core>
#elif __has_include(<eigen3/Eigen/Core>)
#include <eigen3/Eigen/Core>
#else
#error "Eigen/Core not found. Install Eigen headers."
#endif

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace thbvh {

struct PrimitiveBounds {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d m_min = Eigen::Vector3d::Zero();
  Eigen::Vector3d m_max = Eigen::Vector3d::Zero();
  int m_primitive_id = -1;

  Eigen::Vector3d centroid() const { return 0.5 * (m_min + m_max); }
  PrimitiveBounds() = default;

  PrimitiveBounds(
      const Eigen::Vector3d& a,
      const Eigen::Vector3d& b,
      const Eigen::Vector3d& c,
      int primitive_id = -1) {
    m_min = a.cwiseMin(b).cwiseMin(c);
    m_max = a.cwiseMax(b).cwiseMax(c);
    m_primitive_id = primitive_id;
  }
};

class BVHNode4 {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d m_min = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector3d m_max = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
  std::array<BVHNode4*, 4> m_children = {nullptr, nullptr, nullptr, nullptr};
  std::vector<int> m_primitives;

  bool is_leaf() const { return m_children[0] == nullptr; }
};

class FlattenedNode4 {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d m_min = Eigen::Vector3d::Zero();
  Eigen::Vector3d m_max = Eigen::Vector3d::Zero();
  int m_left = 0;
  int m_right = 0;
  int m_child_count = 0;
};

inline void expand_with_primitive(BVHNode4* node, const PrimitiveBounds& primitive) {
  node->m_min = node->m_min.cwiseMin(primitive.m_min);
  node->m_max = node->m_max.cwiseMax(primitive.m_max);
}

inline void compute_bounds_from_range(
    BVHNode4* node,
    const std::vector<PrimitiveBounds>& primitives,
    const std::vector<int>& primitive_ids,
    int begin,
    int end) {
  node->m_min = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  node->m_max = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
  for (int i = begin; i < end; ++i) {
    expand_with_primitive(node, primitives[primitive_ids[i]]);
  }
}

inline int longest_axis(const Eigen::Vector3d& extents) {
  if (extents.x() >= extents.y() && extents.x() >= extents.z()) {
    return 0;
  }
  if (extents.y() >= extents.z()) {
    return 1;
  }
  return 2;
}

inline BVHNode4* build_bvh4_recursive(
    const std::vector<PrimitiveBounds>& primitives,
    std::vector<int>& primitive_ids,
    int begin,
    int end,
    int max_leaf_size) {
  BVHNode4* node = new BVHNode4();
  compute_bounds_from_range(node, primitives, primitive_ids, begin, end);

  const int count = end - begin;
  if (count <= max_leaf_size) {
    node->m_primitives.reserve(count);
    for (int i = begin; i < end; ++i) {
      node->m_primitives.push_back(primitive_ids[i]);
    }
    return node;
  }

  Eigen::Vector3d centroid_min = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector3d centroid_max = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
  for (int i = begin; i < end; ++i) {
    const Eigen::Vector3d c = primitives[primitive_ids[i]].centroid();
    centroid_min = centroid_min.cwiseMin(c);
    centroid_max = centroid_max.cwiseMax(c);
  }

  const Eigen::Vector3d centroid_extent = centroid_max - centroid_min;
  const int axis = longest_axis(centroid_extent);
  if (centroid_extent[axis] <= DBL_EPSILON) {
    node->m_primitives.reserve(count);
    for (int i = begin; i < end; ++i) {
      node->m_primitives.push_back(primitive_ids[i]);
    }
    return node;
  }

  std::sort(
      primitive_ids.begin() + begin,
      primitive_ids.begin() + end,
      [&](int lhs_id, int rhs_id) {
        return primitives[lhs_id].centroid()[axis] < primitives[rhs_id].centroid()[axis];
      });

  const int child_count = std::min(4, count);
  int created_children = 0;
  for (int i = 0; i < child_count; ++i) {
    const int child_begin = begin + (count * i) / child_count;
    const int child_end = begin + (count * (i + 1)) / child_count;
    if (child_end <= child_begin) {
      continue;
    }
    node->m_children[created_children] =
        build_bvh4_recursive(primitives, primitive_ids, child_begin, child_end, max_leaf_size);
    ++created_children;
  }

  if (created_children <= 1) {
    for (int i = 0; i < created_children; ++i) {
      delete node->m_children[i];
      node->m_children[i] = nullptr;
    }
    node->m_primitives.reserve(count);
    for (int i = begin; i < end; ++i) {
      node->m_primitives.push_back(primitive_ids[i]);
    }
  }

  return node;
}

inline std::vector<PrimitiveBounds> build_primitive_bounds(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& faces) {
  if (vertices.cols() != 3 || faces.cols() != 3) {
    throw std::invalid_argument("Vertices/Faces must be Nx3 / Mx3.");
  }

  std::vector<PrimitiveBounds> primitives;
  primitives.reserve(static_cast<size_t>(faces.rows()));
  for (int i = 0; i < faces.rows(); ++i) {
    const int i0 = faces(i, 0);
    const int i1 = faces(i, 1);
    const int i2 = faces(i, 2);
    if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= vertices.rows() || i1 >= vertices.rows() ||
        i2 >= vertices.rows()) {
      throw std::out_of_range("Face index out of range.");
    }
    const Eigen::Vector3d a = vertices.row(i0);
    const Eigen::Vector3d b = vertices.row(i1);
    const Eigen::Vector3d c = vertices.row(i2);
    primitives.emplace_back(a, b, c, i);
  }
  return primitives;
}

inline std::vector<PrimitiveBounds> build_primitive_bounds(
    const std::vector<Eigen::Vector3d>& vertices,
    const std::vector<Eigen::Vector3i>& faces) {
  std::vector<PrimitiveBounds> primitives;
  primitives.reserve(faces.size());
  for (size_t i = 0; i < faces.size(); ++i) {
    const Eigen::Vector3i& face = faces[i];
    const int i0 = face.x();
    const int i1 = face.y();
    const int i2 = face.z();
    if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= static_cast<int>(vertices.size()) ||
        i1 >= static_cast<int>(vertices.size()) || i2 >= static_cast<int>(vertices.size())) {
      throw std::out_of_range("Face index out of range.");
    }
    primitives.emplace_back(vertices[i0], vertices[i1], vertices[i2], static_cast<int>(i));
  }
  return primitives;
}

inline BVHNode4* build_bvh4(const std::vector<PrimitiveBounds>& primitives, int max_leaf_size = 4) {
  if (primitives.empty()) {
    return nullptr;
  }

  std::vector<int> primitive_ids(primitives.size());
  for (int i = 0; i < static_cast<int>(primitives.size()); ++i) {
    primitive_ids[i] = i;
  }
  return build_bvh4_recursive(
      primitives,
      primitive_ids,
      0,
      static_cast<int>(primitive_ids.size()),
      std::max(1, max_leaf_size));
}

inline BVHNode4* build_bvh4(
    const std::vector<Eigen::Vector3d>& vertices,
    const std::vector<Eigen::Vector3i>& faces,
    int max_leaf_size = 4) {
  const std::vector<PrimitiveBounds> primitives = build_primitive_bounds(vertices, faces);
  return build_bvh4(primitives, max_leaf_size);
}

inline BVHNode4* build_bvh4(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& faces,
    int max_leaf_size = 4) {
  const std::vector<PrimitiveBounds> primitives = build_primitive_bounds(vertices, faces);
  return build_bvh4(primitives, max_leaf_size);
}

inline void delete_bvh4(BVHNode4* node) {
  if (node == nullptr) {
    return;
  }
  for (BVHNode4* child : node->m_children) {
    delete_bvh4(child);
  }
  delete node;
}

inline void flatten_bvh4(
    const BVHNode4* root,
    std::vector<FlattenedNode4>& flat_nodes,
    std::vector<int>& primitive_list) {
  flat_nodes.clear();
  primitive_list.clear();
  if (root == nullptr) {
    return;
  }

  struct QueueItem {
    const BVHNode4* node = nullptr;
    int flat_index = -1;
  };

  flat_nodes.push_back(FlattenedNode4());
  std::queue<QueueItem> queue;
  queue.push({root, 0});

  while (!queue.empty()) {
    const QueueItem item = queue.front();
    queue.pop();

    flat_nodes[item.flat_index].m_min = item.node->m_min;
    flat_nodes[item.flat_index].m_max = item.node->m_max;

    if (item.node->is_leaf()) {
      const int begin = static_cast<int>(primitive_list.size());
      primitive_list.insert(
          primitive_list.end(),
          item.node->m_primitives.begin(),
          item.node->m_primitives.end());
      const int end = static_cast<int>(primitive_list.size());

      // Leaf encoding:
      //   m_left  = -(begin + 1)
      //   m_right = end
      // decode begin as (-m_left - 1), then iterate primitive_list[begin, end).
      flat_nodes[item.flat_index].m_left = -(begin + 1);
      flat_nodes[item.flat_index].m_right = end;
      flat_nodes[item.flat_index].m_child_count = 0;
      continue;
    }

    const int child_start = static_cast<int>(flat_nodes.size());
    int child_count = 0;
    for (const BVHNode4* child : item.node->m_children) {
      if (child == nullptr) {
        continue;
      }
      const int child_index = child_start + child_count;
      flat_nodes.push_back(FlattenedNode4());
      queue.push({child, child_index});
      ++child_count;
    }
    // Re-fetch by index because push_back above may reallocate the vector.
    flat_nodes[item.flat_index].m_left = child_start;
    flat_nodes[item.flat_index].m_right = child_start + child_count;
    flat_nodes[item.flat_index].m_child_count = child_count;
  }
}

inline bool is_leaf_encoded(const FlattenedNode4& node) { return node.m_child_count == 0; }

inline int leaf_begin(const FlattenedNode4& node) { return -node.m_left - 1; }

inline int leaf_end(const FlattenedNode4& node) { return node.m_right; }

}  // namespace thbvh