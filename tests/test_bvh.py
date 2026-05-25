#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import trimesh

import torchbvh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load GLB meshes, normalize to unit box, sample outside points on a sphere, "
            "run nearest-point query with torchbvh, and export PLY files."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/root/trellis_toy4k"),
        help="Directory that contains .glb files (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./"),
        help="Directory for exports. Defaults to --input-dir.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4096,
        help="Number of outside query points sampled per mesh.",
    )
    parser.add_argument(
        "--sphere-radius",
        type=float,
        default=1.2,
        help=(
            "Sampling sphere radius in normalized space. "
            "Unit-box mesh lives roughly in [-0.5, 0.5], so > 0.9 is outside."
        ),
    )
    parser.add_argument(
        "--max-leaf-size",
        type=int,
        default=8,
        help="BVH4 max primitive count per leaf node.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for querying (default: cuda).",
    )
    return parser.parse_args()


def find_glb_files(input_dir: Path) -> Iterable[Path]:
    return sorted(input_dir.rglob("*.glb"))


def load_as_single_mesh(glb_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(glb_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise RuntimeError(f"Unsupported geometry type: {type(loaded)}")

    if mesh.faces.shape[1] != 3:
        mesh = mesh.triangulate()

    if mesh.faces.size == 0 or mesh.vertices.size == 0:
        raise RuntimeError("Mesh is empty.")
    return mesh


def normalize_to_unit_box(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    extent = bbox_max - bbox_min
    scale = float(np.max(extent))
    if scale <= 0.0:
        raise RuntimeError("Degenerate mesh with zero extent.")
    vertices_norm = (vertices - center) / scale
    return trimesh.Trimesh(vertices=vertices_norm, faces=mesh.faces, process=False)


def sample_sphere_points(num_points: int, radius: float) -> np.ndarray:
    dirs = np.random.normal(size=(num_points, 3)).astype(np.float32)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    dirs = dirs / norms
    return dirs * radius


def export_query_triangles_ply(
    out_path: str,
    query_points: np.ndarray,
    nearest_points: np.ndarray,
    perturb_eps: float = 2e-3,
) -> None:
    num = query_points.shape[0]
    tri_vertices = np.zeros((num * 3, 3), dtype=np.float32)
    tri_faces = np.zeros((num, 3), dtype=np.int32)

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    for i in range(num):
        src = query_points[i]
        dst = nearest_points[i]
        seg = dst - src
        seg_len = float(np.linalg.norm(seg))

        if seg_len > 1e-12:
            seg_dir = seg / seg_len
            helper = y_axis if abs(float(np.dot(seg_dir, z_axis))) > 0.99 else z_axis
            perp = np.cross(seg_dir, helper)
        else:
            perp = x_axis.copy()

        perp_len = float(np.linalg.norm(perp))
        if perp_len <= 1e-12:
            perp = x_axis.copy()
            perp_len = 1.0
        perp = perp / perp_len

        local_eps = max(perturb_eps, 1e-2 * seg_len)
        src_perturbed = src + perp * local_eps

        base = i * 3
        tri_vertices[base + 0] = src
        tri_vertices[base + 1] = src_perturbed
        tri_vertices[base + 2] = dst
        tri_faces[i] = [base + 0, base + 1, base + 2]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {tri_vertices.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {tri_faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v in tri_vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in tri_faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def run_one_mesh(
    glb_path: Path,
    out_dir: str,
    num_samples: int,
    sphere_radius: float,
    max_leaf_size: int,
    device: torch.device,
) -> None:
    mesh = load_as_single_mesh(glb_path)
    mesh = normalize_to_unit_box(mesh)

    vertices_np = np.asarray(mesh.vertices, dtype=np.float32)
    faces_np = np.asarray(mesh.faces, dtype=np.int32)

    vertices_t = torch.from_numpy(vertices_np).to(device=device, dtype=torch.float32)
    faces_t = torch.from_numpy(faces_np).to(device=device, dtype=torch.int32)

    try:
        bvh = torchbvh._C.build_bvh(vertices_t, faces_t, max_leaf_size=max_leaf_size)
    except Exception as exc:
        raise RuntimeError(f"build_error: {exc}") from exc

    query_np = sample_sphere_points(num_samples, sphere_radius)
    query_t = torch.from_numpy(query_np).to(device=device, dtype=torch.float32)

    # [Q, 8] => [dist2, face_id, closest_x, closest_y, closest_z, bary_u, bary_v, bary_w]
    try:
        result_t = bvh.query(query_t)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except Exception as exc:
        raise RuntimeError(f"query_error: {exc}") from exc

    nearest_t = result_t[:, 2:5]
    nearest_np = nearest_t.detach().cpu().numpy().astype(np.float32)

    out_mesh = os.path.join(out_dir, f"{glb_path.stem}.ply")
    out_query = os.path.join(out_dir, f"{glb_path.stem}_query.ply")
    mesh.export(out_mesh)
    export_query_triangles_ply(out_query, query_np, nearest_np)

    print(f"[OK] {glb_path.name}")
    print(f"     mesh  -> {out_mesh}")
    print(f"     query -> {out_query}")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir if args.output_dir is not None else str(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")

    glb_files = list(find_glb_files(input_dir))
    if not glb_files:
        print(f"No .glb files found under: {input_dir}")
        return

    print(f"Found {len(glb_files)} GLB files under {input_dir}")
    for glb_path in glb_files:
        try:
            run_one_mesh(
                glb_path=glb_path,
                out_dir=output_dir,
                num_samples=args.num_samples,
                sphere_radius=args.sphere_radius,
                max_leaf_size=args.max_leaf_size,
                device=device,
            )
        except Exception as exc:
            msg = str(exc)
            if "build_error:" in msg:
                print(f"[BUILD_FAIL] {glb_path}: {msg}")
            elif "query_error:" in msg:
                print(f"[QUERY_FAIL] {glb_path}: {msg}")
            else:
                print(f"[FAIL] {glb_path}: {msg}")


if __name__ == "__main__":
    main()
