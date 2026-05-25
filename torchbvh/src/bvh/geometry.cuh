#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <limits>

namespace thbvh {

constexpr float kInf = std::numeric_limits<float>::infinity();

struct PointTriangleResult {
  float dist2 = kInf;
  float3 closest = make_float3(0.0f, 0.0f, 0.0f);
  float3 bary = make_float3(0.0f, 0.0f, 0.0f);
};

struct RayTriangleResult {
  bool hit = false;
  float t = kInf;
  float2 bary = make_float2(0.0f, 0.0f);
};

__host__ __device__ inline float3 add3(const float3 a, const float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 sub3(const float3 a, const float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 mul3(const float3 a, const float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float dot3(const float3 a, const float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 cross3(const float3 a, const float3 b) {
  return make_float3(
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x);
}

__host__ __device__ inline float len2_3(const float3 a) { return dot3(a, a); }

__host__ __device__ inline float3 madd3(const float3 a, const float3 b, const float s) {
  return add3(a, mul3(b, s));
}

// Ericson "Real-Time Collision Detection" closest point on triangle.
__device__ inline PointTriangleResult point_triangle_closest(
    const float3 p,
    const float3 a,
    const float3 b,
    const float3 c) {
  PointTriangleResult out;
  const float3 ab = sub3(b, a);
  const float3 ac = sub3(c, a);
  const float3 ap = sub3(p, a);
  const float d1 = dot3(ab, ap);
  const float d2 = dot3(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f) {
    out.closest = a;
    out.bary = make_float3(1.0f, 0.0f, 0.0f);
    out.dist2 = len2_3(sub3(p, a));
    return out;
  }

  const float3 bp = sub3(p, b);
  const float d3 = dot3(ab, bp);
  const float d4 = dot3(ac, bp);
  if (d3 >= 0.0f && d4 <= d3) {
    out.closest = b;
    out.bary = make_float3(0.0f, 1.0f, 0.0f);
    out.dist2 = len2_3(sub3(p, b));
    return out;
  }

  const float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    const float v = d1 / (d1 - d3);
    out.closest = madd3(a, ab, v);
    out.bary = make_float3(1.0f - v, v, 0.0f);
    out.dist2 = len2_3(sub3(p, out.closest));
    return out;
  }

  const float3 cp = sub3(p, c);
  const float d5 = dot3(ab, cp);
  const float d6 = dot3(ac, cp);
  if (d6 >= 0.0f && d5 <= d6) {
    out.closest = c;
    out.bary = make_float3(0.0f, 0.0f, 1.0f);
    out.dist2 = len2_3(sub3(p, c));
    return out;
  }

  const float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    const float w = d2 / (d2 - d6);
    out.closest = madd3(a, ac, w);
    out.bary = make_float3(1.0f - w, 0.0f, w);
    out.dist2 = len2_3(sub3(p, out.closest));
    return out;
  }

  const float va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
    const float3 bc = sub3(c, b);
    const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    out.closest = madd3(b, bc, w);
    out.bary = make_float3(0.0f, 1.0f - w, w);
    out.dist2 = len2_3(sub3(p, out.closest));
    return out;
  }

  const float denom = 1.0f / (va + vb + vc);
  const float v = vb * denom;
  const float w = vc * denom;
  const float u = 1.0f - v - w;
  out.closest = add3(add3(mul3(a, u), mul3(b, v)), mul3(c, w));
  out.bary = make_float3(u, v, w);
  out.dist2 = len2_3(sub3(p, out.closest));
  return out;
}

__device__ inline RayTriangleResult ray_triangle_intersect(
    const float3 ray_o,
    const float3 ray_d,
    const float3 a,
    const float3 b,
    const float3 c,
    float t_min = 1e-6f,
    float t_max = kInf) {
  RayTriangleResult out;
  const float3 e1 = sub3(b, a);
  const float3 e2 = sub3(c, a);
  const float3 pvec = cross3(ray_d, e2);
  const float det = dot3(e1, pvec);
  if (fabsf(det) < 1e-8f) {
    return out;
  }
  const float inv_det = 1.0f / det;
  const float3 tvec = sub3(ray_o, a);
  const float u = dot3(tvec, pvec) * inv_det;
  if (u < 0.0f || u > 1.0f) {
    return out;
  }
  const float3 qvec = cross3(tvec, e1);
  const float v = dot3(ray_d, qvec) * inv_det;
  if (v < 0.0f || (u + v) > 1.0f) {
    return out;
  }
  const float t = dot3(e2, qvec) * inv_det;
  if (t < t_min || t > t_max) {
    return out;
  }
  out.hit = true;
  out.t = t;
  out.bary = make_float2(u, v);
  return out;
}

}  // namespace thbvh
