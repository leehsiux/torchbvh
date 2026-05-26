from importlib import import_module

import torch

from .lbvh import build_bvh_gpu_lbvh

_C = import_module("torchbvh._C")

BVH = _C.BVH


def build_bvh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    max_leaf_size: int = 8,
    *,
    prefer_gpu_lbvh: bool = True,
    gpu_lbvh_max_faces: int = 20_000_000,
) -> BVH:
    use_gpu_lbvh = (
        prefer_gpu_lbvh
        and vertices.is_cuda
        and faces.is_cuda
        and vertices.dtype == torch.float32
        and faces.dtype == torch.int32
        and (gpu_lbvh_max_faces <= 0 or int(faces.shape[0]) <= gpu_lbvh_max_faces)
    )
    if use_gpu_lbvh:
        try:
            return build_bvh_gpu_lbvh(vertices, faces, max_leaf_size=max_leaf_size)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "cuda" in msg:
                # CPU fallback for large meshes / tight GPU memory conditions.
                return _C.build_bvh(vertices.to("cpu"), faces.to("cpu"), max_leaf_size)
            raise

    if vertices.is_cuda:
        vertices = vertices.to("cpu")
    if faces.is_cuda:
        faces = faces.to("cpu")
    return _C.build_bvh(vertices, faces, max_leaf_size)


def point_mesh_query(points: torch.Tensor, bvh: BVH):
    return bvh.query(points)


def ray_mesh_query(ray_origins: torch.Tensor, ray_dirs: torch.Tensor, bvh: BVH):
    return bvh.ray_query(ray_origins, ray_dirs)
