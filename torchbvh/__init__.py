import torch

from . import _C  # noqa: F401


BVH = _C.BVH


def build_bvh(vertices: torch.Tensor, faces: torch.Tensor, max_leaf_size: int = 8) -> BVH:
    return _C.build_bvh(vertices, faces, max_leaf_size)


def point_mesh_query(points: torch.Tensor, bvh: BVH):
    return bvh.query(points)


def ray_mesh_query(ray_origins: torch.Tensor, ray_dirs: torch.Tensor, bvh: BVH):
    return bvh.ray_query(ray_origins, ray_dirs)
