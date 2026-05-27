import math
import torch
from . import _C

def _expand_bits(v: torch.Tensor) -> torch.Tensor:
    # 10-bit input -> 30-bit interleaved output.
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v


def _morton3d_unit(centroids: torch.Tensor) -> torch.Tensor:
    c = torch.clamp(centroids, 0.0, 1.0)
    q = torch.clamp((c * 1023.0).to(torch.int32), 0, 1023)
    xx = _expand_bits(q[:, 0])
    yy = _expand_bits(q[:, 1])
    zz = _expand_bits(q[:, 2])
    return xx | (yy << 1) | (zz << 2)


def _pack_i32_bits_to_f32(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.float32)


def build_bvh_gpu_lbvh(vertices: torch.Tensor, faces: torch.Tensor, max_leaf_size: int = 8):
    if not vertices.is_cuda or not faces.is_cuda:
        raise RuntimeError("GPU LBVH requires CUDA tensors for vertices and faces.")
    if vertices.dtype != torch.float32 or faces.dtype != torch.int32:
        raise RuntimeError("GPU LBVH expects vertices float32 and faces int32.")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError("vertices must be [V,3].")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError("faces must be [F,3].")

    device = vertices.device
    leaf_size = max(1, int(max_leaf_size))
    num_faces = int(faces.shape[0])
    if num_faces == 0:
        raise RuntimeError("Empty faces.")

    tri = faces.to(torch.long)
    tri_vertices = vertices.index_select(0, tri.reshape(-1)).view(num_faces, 3, 3)
    tri_min = torch.amin(tri_vertices, dim=1)
    tri_max = torch.amax(tri_vertices, dim=1)
    centroids = 0.5 * (tri_min + tri_max)

    cmin = torch.amin(centroids, dim=0)
    cmax = torch.amax(centroids, dim=0)
    cextent = torch.clamp(cmax - cmin, min=1e-12)
    cent_unit = (centroids - cmin) / cextent

    morton = _morton3d_unit(cent_unit)
    # Stable sort is dramatically slower on CUDA for large arrays and is not
    # required for LBVH correctness; equal morton-code ties can be arbitrary.
    order = torch.argsort(morton, stable=False)
    primitive_indices = order.to(torch.int32).contiguous()

    tri_min = tri_min.index_select(0, order)
    tri_max = tri_max.index_select(0, order)

    num_leaves = int(math.ceil(num_faces / leaf_size))
    padded = num_leaves * leaf_size

    pad_min = torch.full((padded, 3), float("inf"), dtype=torch.float32, device=device)
    pad_max = torch.full((padded, 3), float("-inf"), dtype=torch.float32, device=device)
    pad_min[:num_faces] = tri_min
    pad_max[:num_faces] = tri_max

    leaf_min = torch.amin(pad_min.view(num_leaves, leaf_size, 3), dim=1)
    leaf_max = torch.amax(pad_max.view(num_leaves, leaf_size, 3), dim=1)
    leaf_begin = (torch.arange(num_leaves, device=device, dtype=torch.int32) * leaf_size).contiguous()
    leaf_end = torch.clamp(leaf_begin + leaf_size, max=num_faces)

    levels: list[dict[str, torch.Tensor]] = [
        {
            "min": leaf_min,
            "max": leaf_max,
            "is_leaf": torch.ones((num_leaves,), dtype=torch.bool, device=device),
            "leaf_begin": leaf_begin,
            "leaf_end": leaf_end,
            "child_start": torch.empty((0,), dtype=torch.int32, device=device),
            "child_count": torch.empty((0,), dtype=torch.int32, device=device),
        }
    ]

    current_min = leaf_min
    current_max = leaf_max
    current_count = num_leaves
    while current_count > 1:
        parent_count = int(math.ceil(current_count / 4))
        pad_n = parent_count * 4

        pmin = torch.full((pad_n, 3), float("inf"), dtype=torch.float32, device=device)
        pmax = torch.full((pad_n, 3), float("-inf"), dtype=torch.float32, device=device)
        pmin[:current_count] = current_min
        pmax[:current_count] = current_max

        parent_min = torch.amin(pmin.view(parent_count, 4, 3), dim=1)
        parent_max = torch.amax(pmax.view(parent_count, 4, 3), dim=1)

        child_start = (torch.arange(parent_count, device=device, dtype=torch.int32) * 4).contiguous()
        child_count = torch.clamp(
            torch.tensor(current_count, device=device, dtype=torch.int32) - child_start,
            min=0,
            max=4,
        ).contiguous()

        levels.append(
            {
                "min": parent_min,
                "max": parent_max,
                "is_leaf": torch.zeros((parent_count,), dtype=torch.bool, device=device),
                "leaf_begin": torch.empty((0,), dtype=torch.int32, device=device),
                "leaf_end": torch.empty((0,), dtype=torch.int32, device=device),
                "child_start": child_start,
                "child_count": child_count,
            }
        )

        current_min = parent_min
        current_max = parent_max
        current_count = parent_count

    levels = list(reversed(levels))
    level_offsets = []
    running = 0
    for lvl in levels:
        level_offsets.append(running)
        running += int(lvl["min"].shape[0])
    total_nodes = running

    node_lower = torch.empty((total_nodes, 4), dtype=torch.float32, device=device)
    node_upper = torch.empty((total_nodes, 4), dtype=torch.float32, device=device)

    for li, lvl in enumerate(levels):
        n = int(lvl["min"].shape[0])
        base = level_offsets[li]
        lo = node_lower[base : base + n]
        hi = node_upper[base : base + n]
        lo[:, :3] = lvl["min"]
        hi[:, :3] = lvl["max"]

        if li == len(levels) - 1:
            left = -(lvl["leaf_begin"] + 1)
            right = lvl["leaf_end"]
        else:
            next_base = level_offsets[li + 1]
            left = (lvl["child_start"] + next_base).to(torch.int32)
            right = (left + lvl["child_count"]).to(torch.int32)

        lo[:, 3] = _pack_i32_bits_to_f32(left.to(torch.int32))
        hi[:, 3] = _pack_i32_bits_to_f32(right.to(torch.int32))

    return _C.build_bvh_from_tensors(
        node_lower.contiguous(),
        node_upper.contiguous(),
        primitive_indices.contiguous(),
        faces.contiguous(),
        vertices.contiguous(),
    )
