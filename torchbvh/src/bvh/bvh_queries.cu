#include "bvh/bvh.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "bvh/geometry.cuh"

namespace thbvh {

namespace {

constexpr int kStackSize = 64;

__device__ inline float3 load3(const float* base, int idx) {
  return make_float3(base[idx * 3 + 0], base[idx * 3 + 1], base[idx * 3 + 2]);
}

__device__ inline int3 load3i(const int32_t* base, int idx) {
  return make_int3(base[idx * 3 + 0], base[idx * 3 + 1], base[idx * 3 + 2]);
}

__device__ inline float point_aabb_dist2(
    const float3 p,
    const float3 bmin,
    const float3 bmax) {
  const float dx = fmaxf(fmaxf(bmin.x - p.x, 0.0f), p.x - bmax.x);
  const float dy = fmaxf(fmaxf(bmin.y - p.y, 0.0f), p.y - bmax.y);
  const float dz = fmaxf(fmaxf(bmin.z - p.z, 0.0f), p.z - bmax.z);
  return dx * dx + dy * dy + dz * dz;
}

__device__ inline bool ray_aabb_intersect(
    const float3 ray_o,
    const float3 inv_d,
    const float3 bmin,
    const float3 bmax,
    float* t_near_out) {
  float tx1 = (bmin.x - ray_o.x) * inv_d.x;
  float tx2 = (bmax.x - ray_o.x) * inv_d.x;
  float tmin = fminf(tx1, tx2);
  float tmax = fmaxf(tx1, tx2);

  float ty1 = (bmin.y - ray_o.y) * inv_d.y;
  float ty2 = (bmax.y - ray_o.y) * inv_d.y;
  tmin = fmaxf(tmin, fminf(ty1, ty2));
  tmax = fminf(tmax, fmaxf(ty1, ty2));

  float tz1 = (bmin.z - ray_o.z) * inv_d.z;
  float tz2 = (bmax.z - ray_o.z) * inv_d.z;
  tmin = fmaxf(tmin, fminf(tz1, tz2));
  tmax = fminf(tmax, fmaxf(tz1, tz2));

  if (tmax < 0.0f || tmin > tmax) {
    return false;
  }
  *t_near_out = fmaxf(0.0f, tmin);
  return true;
}

__device__ inline void compare_swap(float* scores, int* indices, int a, int b) {
  if (scores[a] > scores[b]) {
    const float ts = scores[a];
    scores[a] = scores[b];
    scores[b] = ts;
    const int ti = indices[a];
    indices[a] = indices[b];
    indices[b] = ti;
  }
}

// Sorting network for 4 items.
__device__ inline void sort4(float* scores, int* indices) {
  compare_swap(scores, indices, 0, 1);
  compare_swap(scores, indices, 2, 3);
  compare_swap(scores, indices, 0, 2);
  compare_swap(scores, indices, 1, 3);
  compare_swap(scores, indices, 1, 2);
}

__global__ void point_mesh_query_kernel(
    const float* points,
    int64_t query_count,
    const float* node_min,
    const float* node_max,
    const int32_t* node_left,
    const int32_t* node_right,
    const int32_t* node_child_count,
    const int32_t* primitive_indices,
    const int32_t* faces,
    const float* vertices,
    int32_t node_count,
    int32_t primitive_count,
    int32_t face_count,
    int32_t vertex_count,
    float* out) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= query_count) {
    return;
  }

  const float3 query = load3(points, idx);
  float best_dist2 = kInf;
  int best_face = -1;
  float3 best_closest = make_float3(0.0f, 0.0f, 0.0f);
  float3 best_bary = make_float3(0.0f, 0.0f, 0.0f);

  int stack[kStackSize];
  int stack_size = 0;
  stack[stack_size++] = 0;

  while (stack_size > 0) {
    const int node_idx = stack[--stack_size];
    if (node_idx < 0 || node_idx >= node_count) {
      continue;
    }
    const float3 bmin = load3(node_min, node_idx);
    const float3 bmax = load3(node_max, node_idx);
    if (point_aabb_dist2(query, bmin, bmax) > best_dist2) {
      continue;
    }

    const int child_count = node_child_count[node_idx];
    if (child_count == 0) {
      const int begin = max(0, -node_left[node_idx] - 1);
      const int end = min(static_cast<int>(primitive_count), node_right[node_idx]);
      if (end < begin) {
        continue;
      }
      for (int p = begin; p < end; ++p) {
        const int primitive_id = primitive_indices[p];
        if (primitive_id < 0 || primitive_id >= face_count) {
          continue;
        }
        const int3 tri = load3i(faces, primitive_id);
        if (tri.x < 0 || tri.y < 0 || tri.z < 0 || tri.x >= vertex_count || tri.y >= vertex_count ||
            tri.z >= vertex_count) {
          continue;
        }
        const float3 a = load3(vertices, tri.x);
        const float3 b = load3(vertices, tri.y);
        const float3 c = load3(vertices, tri.z);
        const PointTriangleResult result = point_triangle_closest(query, a, b, c);
        if (result.dist2 < best_dist2) {
          best_dist2 = result.dist2;
          best_face = primitive_id;
          best_closest = result.closest;
          best_bary = result.bary;
        }
      }
      continue;
    }

    float scores[4] = {kInf, kInf, kInf, kInf};
    int children[4] = {-1, -1, -1, -1};
    const int child_begin = node_left[node_idx];
    for (int c = 0; c < child_count && c < 4; ++c) {
      const int child_idx = child_begin + c;
      if (child_idx < 0 || child_idx >= node_count) {
        continue;
      }
      const float3 cbmin = load3(node_min, child_idx);
      const float3 cbmax = load3(node_max, child_idx);
      scores[c] = point_aabb_dist2(query, cbmin, cbmax);
      children[c] = child_idx;
    }

    sort4(scores, children);
    for (int c = 3; c >= 0; --c) {
      const int child_idx = children[c];
      if (child_idx < 0 || !isfinite(scores[c]) || scores[c] > best_dist2) {
        continue;
      }
      if (stack_size < kStackSize) {
        stack[stack_size++] = child_idx;
      }
    }
  }

  const int out_offset = idx * 8;
  out[out_offset + 0] = best_dist2;
  out[out_offset + 1] = static_cast<float>(best_face);
  out[out_offset + 2] = best_closest.x;
  out[out_offset + 3] = best_closest.y;
  out[out_offset + 4] = best_closest.z;
  out[out_offset + 5] = best_bary.x;
  out[out_offset + 6] = best_bary.y;
  out[out_offset + 7] = best_bary.z;
}

__global__ void ray_mesh_query_kernel(
    const float* ray_origins,
    const float* ray_dirs,
    int64_t query_count,
    const float* node_min,
    const float* node_max,
    const int32_t* node_left,
    const int32_t* node_right,
    const int32_t* node_child_count,
    const int32_t* primitive_indices,
    const int32_t* faces,
    const float* vertices,
    int32_t node_count,
    int32_t primitive_count,
    int32_t face_count,
    int32_t vertex_count,
    float* out) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= query_count) {
    return;
  }

  const float3 ray_o = load3(ray_origins, idx);
  const float3 ray_d = load3(ray_dirs, idx);
  const float3 inv_d =
      make_float3(1.0f / ray_d.x, 1.0f / ray_d.y, 1.0f / ray_d.z);

  float best_t = kInf;
  int best_face = -1;
  float best_u = 0.0f;
  float best_v = 0.0f;

  int stack[kStackSize];
  int stack_size = 0;
  stack[stack_size++] = 0;

  while (stack_size > 0) {
    const int node_idx = stack[--stack_size];
    if (node_idx < 0 || node_idx >= node_count) {
      continue;
    }
    const float3 bmin = load3(node_min, node_idx);
    const float3 bmax = load3(node_max, node_idx);
    float t_near = 0.0f;
    if (!ray_aabb_intersect(ray_o, inv_d, bmin, bmax, &t_near) || t_near > best_t) {
      continue;
    }

    const int child_count = node_child_count[node_idx];
    if (child_count == 0) {
      const int begin = max(0, -node_left[node_idx] - 1);
      const int end = min(static_cast<int>(primitive_count), node_right[node_idx]);
      if (end < begin) {
        continue;
      }
      for (int p = begin; p < end; ++p) {
        const int primitive_id = primitive_indices[p];
        if (primitive_id < 0 || primitive_id >= face_count) {
          continue;
        }
        const int3 tri = load3i(faces, primitive_id);
        if (tri.x < 0 || tri.y < 0 || tri.z < 0 || tri.x >= vertex_count || tri.y >= vertex_count ||
            tri.z >= vertex_count) {
          continue;
        }
        const float3 a = load3(vertices, tri.x);
        const float3 b = load3(vertices, tri.y);
        const float3 c = load3(vertices, tri.z);
        const RayTriangleResult hit = ray_triangle_intersect(ray_o, ray_d, a, b, c, 1e-6f, best_t);
        if (hit.hit && hit.t < best_t) {
          best_t = hit.t;
          best_face = primitive_id;
          best_u = hit.bary.x;
          best_v = hit.bary.y;
        }
      }
      continue;
    }

    float scores[4] = {kInf, kInf, kInf, kInf};
    int children[4] = {-1, -1, -1, -1};
    const int child_begin = node_left[node_idx];
    for (int c = 0; c < child_count && c < 4; ++c) {
      const int child_idx = child_begin + c;
      if (child_idx < 0 || child_idx >= node_count) {
        continue;
      }
      const float3 cbmin = load3(node_min, child_idx);
      const float3 cbmax = load3(node_max, child_idx);
      float t_child_near = 0.0f;
      if (ray_aabb_intersect(ray_o, inv_d, cbmin, cbmax, &t_child_near)) {
        scores[c] = t_child_near;
        children[c] = child_idx;
      }
    }

    sort4(scores, children);
    for (int c = 3; c >= 0; --c) {
      const int child_idx = children[c];
      if (child_idx < 0 || !isfinite(scores[c]) || scores[c] > best_t) {
        continue;
      }
      if (stack_size < kStackSize) {
        stack[stack_size++] = child_idx;
      }
    }
  }

  const bool hit = best_face >= 0;
  const float w = 1.0f - best_u - best_v;
  const float3 hit_point = hit ? add3(ray_o, mul3(ray_d, best_t)) : make_float3(0.0f, 0.0f, 0.0f);
  const int out_offset = idx * 9;
  out[out_offset + 0] = hit ? 1.0f : 0.0f;
  out[out_offset + 1] = hit ? best_t : kInf;
  out[out_offset + 2] = static_cast<float>(best_face);
  out[out_offset + 3] = hit ? best_u : 0.0f;
  out[out_offset + 4] = hit ? best_v : 0.0f;
  out[out_offset + 5] = hit ? w : 0.0f;
  out[out_offset + 6] = hit_point.x;
  out[out_offset + 7] = hit_point.y;
  out[out_offset + 8] = hit_point.z;
}

}  // namespace

torch::Tensor point_mesh_query_cuda(
    const torch::Tensor& points,
    const torch::Tensor& node_min,
    const torch::Tensor& node_max,
    const torch::Tensor& node_left,
    const torch::Tensor& node_right,
    const torch::Tensor& node_child_count,
    const torch::Tensor& primitive_indices,
    const torch::Tensor& faces,
    const torch::Tensor& vertices) {
  const int64_t q = points.size(0);
  auto out = torch::zeros({q, 8}, points.options().dtype(torch::kFloat32));
  if (q == 0) {
    return out;
  }

  const int threads = 256;
  const int blocks = static_cast<int>((q + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  point_mesh_query_kernel<<<blocks, threads, 0, stream>>>(
      points.data_ptr<float>(),
      q,
      node_min.data_ptr<float>(),
      node_max.data_ptr<float>(),
      node_left.data_ptr<int32_t>(),
      node_right.data_ptr<int32_t>(),
      node_child_count.data_ptr<int32_t>(),
      primitive_indices.data_ptr<int32_t>(),
      faces.data_ptr<int32_t>(),
      vertices.data_ptr<float>(),
      static_cast<int32_t>(node_left.size(0)),
      static_cast<int32_t>(primitive_indices.size(0)),
      static_cast<int32_t>(faces.size(0)),
      static_cast<int32_t>(vertices.size(0)),
      out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor ray_mesh_query_cuda(
    const torch::Tensor& ray_origins,
    const torch::Tensor& ray_dirs,
    const torch::Tensor& node_min,
    const torch::Tensor& node_max,
    const torch::Tensor& node_left,
    const torch::Tensor& node_right,
    const torch::Tensor& node_child_count,
    const torch::Tensor& primitive_indices,
    const torch::Tensor& faces,
    const torch::Tensor& vertices) {
  const int64_t q = ray_origins.size(0);
  auto out = torch::zeros({q, 9}, ray_origins.options().dtype(torch::kFloat32));
  if (q == 0) {
    return out;
  }

  const int threads = 256;
  const int blocks = static_cast<int>((q + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  ray_mesh_query_kernel<<<blocks, threads, 0, stream>>>(
      ray_origins.data_ptr<float>(),
      ray_dirs.data_ptr<float>(),
      q,
      node_min.data_ptr<float>(),
      node_max.data_ptr<float>(),
      node_left.data_ptr<int32_t>(),
      node_right.data_ptr<int32_t>(),
      node_child_count.data_ptr<int32_t>(),
      primitive_indices.data_ptr<int32_t>(),
      faces.data_ptr<int32_t>(),
      vertices.data_ptr<float>(),
      static_cast<int32_t>(node_left.size(0)),
      static_cast<int32_t>(primitive_indices.size(0)),
      static_cast<int32_t>(faces.size(0)),
      static_cast<int32_t>(vertices.size(0)),
      out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace thbvh
