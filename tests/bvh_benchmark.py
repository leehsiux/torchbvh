#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
import numpy as np
import torch
import trimesh
from torch.utils.cpp_extension import load_inline

import torchbvh

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark BVH build/query/precision on /root/trellis_toy4k comparing "
            "igl (CPU ground truth), cubvh, warp-lang (if available), and torchbvh."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("/root/trellis_toy4k"))
    parser.add_argument(
        "--extra-glob",
        type=str,
        default="/mnt/pfs/users/lixiu/data/*.glb",
        help="Additional glob for extra meshes (default includes large meshes in $HOME/data/*.glb).",
    )
    parser.add_argument("--libigl-dir", type=Path, default=Path("/mnt/pfs/users/lixiu/dev/libigl"))
    parser.add_argument("--output-dir", type=Path, default=Path("./benchmark_out"))
    parser.add_argument("--max-meshes", type=int, default=20, help="Max meshes to benchmark; <=0 means all.")
    parser.add_argument("--num-query-points", type=int, default=8192)
    parser.add_argument("--sphere-radius", type=float, default=1.2)
    parser.add_argument("--max-leaf-size", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=3, help="Timed query repeats.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup query runs.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ours-prefer-gpu-lbvh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer torchbvh GPU LBVH build path when possible.",
    )
    parser.add_argument(
        "--ours-gpu-lbvh-max-faces",
        type=int,
        default=200_000_000,
        help="Face-count guard for GPU LBVH; larger meshes fallback to CPU build.",
    )
    return parser.parse_args()


def list_glb_files(root: Path, extra_glob: str, max_meshes: int) -> list[Path]:
    files: list[Path] = []
    files.extend(root.rglob("*.glb"))
    if extra_glob:
        expanded = os.path.expandvars(os.path.expanduser(extra_glob))
        files.extend(Path(p).resolve() for p in glob.glob(expanded))

    # Deduplicate while preserving deterministic ordering.
    unique = sorted({p.resolve() for p in files})
    if max_meshes <= 0:
        return unique
    return unique[:max_meshes]


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path)
    if loaded.faces.shape[1] != 3:
        loaded = loaded.triangulate()
    if loaded.faces.size == 0 or loaded.vertices.size == 0:
        raise RuntimeError("Empty mesh")
    return loaded


def normalize_mesh(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int32)
    bmin = v.min(axis=0)
    bmax = v.max(axis=0)
    center = 0.5 * (bmin + bmax)
    scale = float(np.max(bmax - bmin))
    if scale <= 0.0:
        raise RuntimeError("Degenerate mesh")
    v = (v - center) / scale
    return v, f


def sample_query_points(num_points: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    d = rng.normal(size=(num_points, 3)).astype(np.float32)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    d = d / n
    return d * radius


def rmse(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(x))))


def to_float64_matrix(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


def to_int32_matrix(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.int32)


def load_igl_jit_module(libigl_dir: Path):
    include_dir = libigl_dir / "include"
    if not include_dir.exists():
        raise FileNotFoundError(f"libigl include dir not found: {include_dir}")

    torch_eigen = Path(torch.__file__).resolve().parent / "include" / "third_party" / "eigen3"
    extra_includes = [str(include_dir), "/usr/include/eigen3"]
    if torch_eigen.exists():
        extra_includes.append(str(torch_eigen))

    cpp_src = r"""
    #include <pybind11/eigen.h>
    #include <pybind11/pybind11.h>
    #include <igl/AABB.h>
    #include <Eigen/Core>
    #include <tuple>

    namespace py = pybind11;

    struct IglBVH {
      Eigen::MatrixXd V;
      Eigen::MatrixXi F;
      igl::AABB<Eigen::MatrixXd, 3> tree;

      IglBVH(const Eigen::Ref<const Eigen::MatrixXd>& v,
             const Eigen::Ref<const Eigen::MatrixXi>& f)
          : V(v), F(f) {
        tree.init(V, F);
      }

      std::tuple<Eigen::VectorXd, Eigen::VectorXi, Eigen::MatrixXd> query(
          const Eigen::Ref<const Eigen::MatrixXd>& P) const {
        Eigen::VectorXd sqrD;
        Eigen::VectorXi I;
        Eigen::MatrixXd C;
        tree.squared_distance(V, F, P, sqrD, I, C);
        return std::make_tuple(sqrD, I, C);
      }
    };

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      py::class_<IglBVH>(m, "IglBVH")
          .def(py::init<const Eigen::Ref<const Eigen::MatrixXd>&,
                        const Eigen::Ref<const Eigen::MatrixXi>&>())
          .def("query", &IglBVH::query);
    }
    """

    return load_inline(
        name="igl_bvh_jit",
        cpp_sources=[cpp_src],
        extra_include_paths=extra_includes,
        functions=None,
        extra_cflags=["-O3", "-std=c++17"],
        with_cuda=False,
        verbose=False,
    )


@dataclass
class BackendResult:
    build_ms: float
    query_ms: float
    distances: np.ndarray
    nearest_points: np.ndarray
    available: bool = True
    reason: str = ""


class BackendBase:
    name = "base"

    def run(self, vertices: np.ndarray, faces: np.ndarray, queries: np.ndarray, repeat: int, warmup: int) -> BackendResult:
        raise NotImplementedError


class IglBackend(BackendBase):
    name = "igl_cpu"

    def __init__(self, libigl_module):
        self.libigl_module = libigl_module

    def run(self, vertices: np.ndarray, faces: np.ndarray, queries: np.ndarray, repeat: int, warmup: int) -> BackendResult:
        v64 = to_float64_matrix(vertices)
        f32 = to_int32_matrix(faces)
        q64 = to_float64_matrix(queries)

        t0 = perf_counter()
        bvh = self.libigl_module.IglBVH(v64, f32)
        build_ms = (perf_counter() - t0) * 1000.0

        for _ in range(max(0, warmup)):
            _ = bvh.query(q64)

        timings = []
        sqrD = None
        closest = None
        for _ in range(max(1, repeat)):
            t0 = perf_counter()
            sqrD, _, closest = bvh.query(q64)
            timings.append((perf_counter() - t0) * 1000.0)

        distances = np.sqrt(np.asarray(sqrD, dtype=np.float64))
        nearest_points = np.asarray(closest, dtype=np.float64)
        return BackendResult(
            build_ms=build_ms,
            query_ms=float(np.mean(timings)),
            distances=distances,
            nearest_points=nearest_points,
        )


class OursBackend(BackendBase):
    name = "torchbvh_cuda"

    def __init__(
        self,
        device: torch.device,
        max_leaf_size: int,
        prefer_gpu_lbvh: bool,
        gpu_lbvh_max_faces: int,
    ):
        self.device = device
        self.max_leaf_size = max_leaf_size
        self.prefer_gpu_lbvh = prefer_gpu_lbvh
        self.gpu_lbvh_max_faces = gpu_lbvh_max_faces

    def run(self, vertices: np.ndarray, faces: np.ndarray, queries: np.ndarray, repeat: int, warmup: int) -> BackendResult:
        if self.device.type != "cuda":
            return BackendResult(0.0, 0.0, np.array([]), np.array([]), available=False, reason="requires CUDA")

        v = torch.from_numpy(vertices).to(self.device, dtype=torch.float32)
        f = torch.from_numpy(faces).to(self.device, dtype=torch.int32)
        q = torch.from_numpy(queries).to(self.device, dtype=torch.float32)

        torch.cuda.synchronize(self.device)
        t0 = perf_counter()
        bvh = torchbvh.build_bvh(
            v,
            f,
            self.max_leaf_size,
            prefer_gpu_lbvh=self.prefer_gpu_lbvh,
            gpu_lbvh_max_faces=self.gpu_lbvh_max_faces,
        )
        torch.cuda.synchronize(self.device)
        build_ms = (perf_counter() - t0) * 1000.0

        for _ in range(max(0, warmup)):
            _ = bvh.query(q)
        torch.cuda.synchronize(self.device)

        timings = []
        out = None
        for _ in range(max(1, repeat)):
            t0 = perf_counter()
            out = bvh.query(q)
            torch.cuda.synchronize(self.device)
            timings.append((perf_counter() - t0) * 1000.0)

        dist = torch.sqrt(torch.clamp(out[:, 0], min=0.0)).detach().cpu().numpy().astype(np.float64)
        nearest_points = out[:, 2:5].detach().cpu().numpy().astype(np.float64)
        return BackendResult(
            build_ms=build_ms,
            query_ms=float(np.mean(timings)),
            distances=dist,
            nearest_points=nearest_points,
        )


class CubvhBackend(BackendBase):
    name = "cubvh_cuda"

    def __init__(self, device: torch.device):
        self.device = device
        try:
            import cubvh  # type: ignore

            self.cubvh = cubvh
            self.available = True
        except Exception as exc:
            self.available = False
            self.reason = f"import failed: {exc}"

    def run(self, vertices: np.ndarray, faces: np.ndarray, queries: np.ndarray, repeat: int, warmup: int) -> BackendResult:
        if not self.available:
            return BackendResult(0.0, 0.0, np.array([]), np.array([]), available=False, reason=self.reason)
        if self.device.type != "cuda":
            return BackendResult(0.0, 0.0, np.array([]), np.array([]), available=False, reason="requires CUDA")

        v = torch.from_numpy(vertices).to(self.device, dtype=torch.float32)
        f = torch.from_numpy(faces).to(self.device, dtype=torch.int32)
        q = torch.from_numpy(queries).to(self.device, dtype=torch.float32)

        torch.cuda.synchronize(self.device)
        t0 = perf_counter()
        bvh = self.cubvh.cuBVH(v, f)
        torch.cuda.synchronize(self.device)
        build_ms = (perf_counter() - t0) * 1000.0

        for _ in range(max(0, warmup)):
            _ = bvh.unsigned_distance(q, return_uvw=False)
        torch.cuda.synchronize(self.device)

        timings = []
        out_dist = None
        out_face = None
        out_aux = None
        for _ in range(max(1, repeat)):
            t0 = perf_counter()
            out = bvh.unsigned_distance(q, return_uvw=True)
            torch.cuda.synchronize(self.device)
            timings.append((perf_counter() - t0) * 1000.0)
            if isinstance(out, tuple):
                out_dist, out_face, out_aux = out
            else:
                out_dist = out
                out_face = None
                out_aux = None

        dist = torch.as_tensor(out_dist).detach().cpu().numpy().astype(np.float64)
        nearest_points = np.array([])
        if out_aux is not None:
            aux = torch.as_tensor(out_aux).detach().cpu().numpy().astype(np.float64)
            if out_face is not None and aux.ndim == 2 and aux.shape[1] == 3:
                # cubvh returns barycentric coordinates (uvw) with face ids.
                # Reconstruct closest point in world coordinates to compare to igl.
                face_idx_raw = torch.as_tensor(out_face).detach().cpu().numpy().astype(np.int64)
                valid = face_idx_raw >= 0
                face_idx = np.clip(face_idx_raw, 0, faces.shape[0] - 1)
                tri = faces[face_idx]  # [Q,3]
                tri_v = vertices[tri]  # [Q,3,3]
                u = aux[:, 0:1]
                v = aux[:, 1:2]
                w = aux[:, 2:3]
                nearest_points = (
                    tri_v[:, 0, :].astype(np.float64) * u
                    + tri_v[:, 1, :].astype(np.float64) * v
                    + tri_v[:, 2, :].astype(np.float64) * w
                )
                if not np.all(valid):
                    nearest_points[~valid] = np.nan
            elif aux.ndim == 2 and aux.shape[1] == 3:
                # Fallback for variants that return closest-point xyz directly.
                nearest_points = aux
        return BackendResult(
            build_ms=build_ms,
            query_ms=float(np.mean(timings)),
            distances=dist,
            nearest_points=nearest_points,
        )


class WarpBackend(BackendBase):
    name = "warp_cuda"

    def __init__(self, device: torch.device):
        self.device = device
        try:
            import warp as wp  # type: ignore

            self.wp = wp
            self.available = True
        except Exception as exc:
            self.available = False
            self.reason = f"import failed: {exc}"

    def run(self, vertices: np.ndarray, faces: np.ndarray, queries: np.ndarray, repeat: int, warmup: int) -> BackendResult:
        if not self.available:
            return BackendResult(0.0, 0.0, np.array([]), np.array([]), available=False, reason=self.reason)
        if self.device.type != "cuda":
            return BackendResult(0.0, 0.0, np.array([]), np.array([]), available=False, reason="requires CUDA")

        wp = self.wp
        try:
            wp.init()
            v = wp.array(vertices, dtype=wp.vec3, device="cuda")
            idx = wp.array(faces.reshape(-1), dtype=wp.int32, device="cuda")
            q = wp.array(queries, dtype=wp.vec3, device="cuda")
            sqr = wp.zeros(shape=queries.shape[0], dtype=wp.float32, device="cuda")
            fid = wp.zeros(shape=queries.shape[0], dtype=wp.int32, device="cuda")
            u = wp.zeros(shape=queries.shape[0], dtype=wp.float32, device="cuda")
            vv = wp.zeros(shape=queries.shape[0], dtype=wp.float32, device="cuda")
            cp = wp.zeros(shape=queries.shape[0], dtype=wp.vec3, device="cuda")

            @wp.kernel
            def query_kernel(
                mesh_id: wp.uint64,
                points: wp.array(dtype=wp.vec3),
                max_dist: float,
                out_sqr: wp.array(dtype=float),
                out_face: wp.array(dtype=int),
                out_u: wp.array(dtype=float),
                out_v: wp.array(dtype=float),
                out_cp: wp.array(dtype=wp.vec3),
            ):
                tid = wp.tid()
                face = int(0)
                uu = float(0.0)
                vv_ = float(0.0)
                sign = float(0.0)
                p = points[tid]
                hit = wp.mesh_query_point(mesh_id, p, max_dist, sign, face, uu, vv_)
                if hit:
                    cp = wp.mesh_eval_position(mesh_id, face, uu, vv_)
                    d = p - cp
                    out_sqr[tid] = wp.dot(d, d)
                    out_face[tid] = face
                    out_u[tid] = uu
                    out_v[tid] = vv_
                    out_cp[tid] = cp
                else:
                    out_sqr[tid] = max_dist * max_dist
                    out_face[tid] = -1
                    out_u[tid] = 0.0
                    out_v[tid] = 0.0
                    out_cp[tid] = p

            t0 = perf_counter()
            mesh = wp.Mesh(points=v, indices=idx)
            wp.synchronize()
            build_ms = (perf_counter() - t0) * 1000.0

            for _ in range(max(0, warmup)):
                wp.launch(
                    query_kernel,
                    dim=queries.shape[0],
                    inputs=[mesh.id, q, 10.0, sqr, fid, u, vv, cp],
                    device="cuda",
                )
                wp.synchronize()

            timings = []
            for _ in range(max(1, repeat)):
                t0 = perf_counter()
                wp.launch(
                    query_kernel,
                    dim=queries.shape[0],
                    inputs=[mesh.id, q, 10.0, sqr, fid, u, vv, cp],
                    device="cuda",
                )
                wp.synchronize()
                timings.append((perf_counter() - t0) * 1000.0)

            sqr_np = wp.to_torch(sqr).detach().cpu().numpy().astype(np.float64)
            dist = np.sqrt(np.clip(sqr_np, 0.0, None))
            cp_np = wp.to_torch(cp).detach().cpu().numpy().astype(np.float64)
            return BackendResult(
                build_ms=build_ms,
                query_ms=float(np.mean(timings)),
                distances=dist,
                nearest_points=cp_np,
            )
        except Exception:
            return BackendResult(
                0.0,
                0.0,
                np.array([]),
                np.array([]),
                available=False,
                reason="runtime error: " + traceback.format_exc(limit=1).strip().replace("\n", " | "),
            )


def compute_precision(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    if pred.size == 0 or gt.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "max_abs": float("nan")}
    n = min(pred.shape[0], gt.shape[0])
    err = np.abs(pred[:n] - gt[:n])
    return {"mae": float(np.mean(err)), "rmse": rmse(pred[:n] - gt[:n]), "max_abs": float(np.max(err))}


def compute_point_precision(pred_points: np.ndarray, gt_points: np.ndarray) -> dict[str, float]:
    if pred_points.size == 0 or gt_points.size == 0:
        return {"point_mae": float("nan"), "point_rmse": float("nan"), "point_max_abs": float("nan")}
    n = min(pred_points.shape[0], gt_points.shape[0])
    delta = pred_points[:n] - gt_points[:n]
    l2 = np.linalg.norm(delta, axis=1)
    l2 = l2[np.isfinite(l2)]
    if l2.size == 0:
        return {"point_mae": float("nan"), "point_rmse": float("nan"), "point_max_abs": float("nan")}
    return {
        "point_mae": float(np.mean(l2)),
        "point_rmse": rmse(l2),
        "point_max_abs": float(np.max(l2)),
    }


def draw_metric_curve(rows: list[dict[str, Any]], metric: str, y_label: str, title: str, out_path: Path) -> None:
    # Keep igl as numerical reference only; omit it from speed/accuracy plots.
    backends = sorted({row["backend"] for row in rows if row["backend"] != "igl_cpu"})
    plt.figure(figsize=(10, 6))
    drew_any = False
    for backend in backends:
        pts: list[tuple[int, float]] = []
        for row in rows:
            if row["backend"] != backend:
                continue
            if not row["available"]:
                continue
            x = row["mesh_face_count"]
            y = row[metric]
            if x is None or y is None:
                continue
            if not np.isfinite(y):
                continue
            pts.append((int(x), float(y)))
        pts.sort(key=lambda t: t[0])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=4, label=backend)
        drew_any = True

    if not drew_any:
        return

    plt.xlabel("Mesh face count")
    plt.xscale("log")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def build_leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    backends = sorted({row["backend"] for row in rows})
    leaderboard: list[dict[str, Any]] = []
    for backend in backends:
        valid = [row for row in rows if row["backend"] == backend and row["available"]]
        if not valid:
            leaderboard.append(
                {
                    "backend": backend,
                    "num_meshes": 0,
                    "avg_build_ms": float("nan"),
                    "avg_query_ms": float("nan"),
                    "avg_mae_vs_igl": float("nan"),
                    "avg_rmse_vs_igl": float("nan"),
                    "avg_max_abs_vs_igl": float("nan"),
                    "avg_point_mae_vs_igl": float("nan"),
                    "avg_point_rmse_vs_igl": float("nan"),
                    "avg_point_max_abs_vs_igl": float("nan"),
                }
            )
            continue

        def avg_metric(key: str) -> float:
            vals = [float(r[key]) for r in valid if np.isfinite(float(r[key]))]
            if not vals:
                return float("nan")
            return float(np.mean(vals))

        leaderboard.append(
            {
                "backend": backend,
                "num_meshes": len(valid),
                "avg_build_ms": avg_metric("build_ms"),
                "avg_query_ms": avg_metric("query_ms"),
                "avg_mae_vs_igl": avg_metric("mae_vs_igl"),
                "avg_rmse_vs_igl": avg_metric("rmse_vs_igl"),
                "avg_max_abs_vs_igl": avg_metric("max_abs_vs_igl"),
                "avg_point_mae_vs_igl": avg_metric("point_mae_vs_igl"),
                "avg_point_rmse_vs_igl": avg_metric("point_rmse_vs_igl"),
                "avg_point_max_abs_vs_igl": avg_metric("point_max_abs_vs_igl"),
            }
        )

    def rank_key(metric: str):
        ranked = [r for r in leaderboard if np.isfinite(r[metric])]
        ranked.sort(key=lambda r: r[metric])
        rank_map = {r["backend"]: idx + 1 for idx, r in enumerate(ranked)}
        for r in leaderboard:
            r[f"rank_{metric}"] = rank_map.get(r["backend"], None)

    rank_key("avg_build_ms")
    rank_key("avg_query_ms")
    rank_key("avg_rmse_vs_igl")
    rank_key("avg_point_rmse_vs_igl")
    return leaderboard


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    glb_files = list_glb_files(args.input_dir, args.extra_glob, args.max_meshes)
    if not glb_files:
        raise RuntimeError(f"No GLB files found in {args.input_dir}")

    print(f"Compiling/loading libigl JIT from {args.libigl_dir} ...")
    igl_mod = load_igl_jit_module(args.libigl_dir)

    backends: list[BackendBase] = [
        IglBackend(igl_mod),
        CubvhBackend(device),
        WarpBackend(device),
        OursBackend(
            device,
            args.max_leaf_size,
            args.ours_prefer_gpu_lbvh,
            args.ours_gpu_lbvh_max_faces,
        ),
    ]

    rows: list[dict[str, Any]] = []
    for mesh_path in glb_files:
        print(f"\n=== {mesh_path.name} ===")
        mesh = normalize_mesh(load_mesh(mesh_path))
        vertices, faces = mesh
        face_count = int(faces.shape[0])
        queries = sample_query_points(args.num_query_points, args.sphere_radius, rng)

        # Ground truth from igl
        gt_result = backends[0].run(vertices, faces, queries, repeat=args.repeat, warmup=args.warmup)
        gt = gt_result.distances
        gt_points = gt_result.nearest_points
        rows.append(
            {
                "mesh": mesh_path.name,
                "mesh_face_count": face_count,
                "backend": backends[0].name,
                "available": True,
                "reason": "",
                "build_ms": gt_result.build_ms,
                "query_ms": gt_result.query_ms,
                "mae_vs_igl": 0.0,
                "rmse_vs_igl": 0.0,
                "max_abs_vs_igl": 0.0,
                "point_mae_vs_igl": 0.0,
                "point_rmse_vs_igl": 0.0,
                "point_max_abs_vs_igl": 0.0,
            }
        )
        print(f"{backends[0].name:12s} build={gt_result.build_ms:8.3f}ms query={gt_result.query_ms:8.3f}ms")

        for backend in backends[1:]:
            result = backend.run(vertices, faces, queries, repeat=args.repeat, warmup=args.warmup)
            if not result.available:
                rows.append(
                    {
                        "mesh": mesh_path.name,
                        "mesh_face_count": face_count,
                        "backend": backend.name,
                        "available": False,
                        "reason": result.reason,
                        "build_ms": float("nan"),
                        "query_ms": float("nan"),
                        "mae_vs_igl": float("nan"),
                        "rmse_vs_igl": float("nan"),
                        "max_abs_vs_igl": float("nan"),
                        "point_mae_vs_igl": float("nan"),
                        "point_rmse_vs_igl": float("nan"),
                        "point_max_abs_vs_igl": float("nan"),
                    }
                )
                print(f"{backend.name:12s} unavailable: {result.reason}")
                continue

            p = compute_precision(result.distances, gt)
            pp = compute_point_precision(result.nearest_points, gt_points)
            rows.append(
                {
                    "mesh": mesh_path.name,
                    "mesh_face_count": face_count,
                    "backend": backend.name,
                    "available": True,
                    "reason": "",
                    "build_ms": result.build_ms,
                    "query_ms": result.query_ms,
                    "mae_vs_igl": p["mae"],
                    "rmse_vs_igl": p["rmse"],
                    "max_abs_vs_igl": p["max_abs"],
                    "point_mae_vs_igl": pp["point_mae"],
                    "point_rmse_vs_igl": pp["point_rmse"],
                    "point_max_abs_vs_igl": pp["point_max_abs"],
                }
            )
            print(
                f"{backend.name:12s} build={result.build_ms:8.3f}ms "
                f"query={result.query_ms:8.3f}ms "
                f"dist_rmse={p['rmse']:.6e} point_rmse={pp['point_rmse']:.6e} "
                f"dist_max={p['max_abs']:.6e} point_max={pp['point_max_abs']:.6e}"
            )

    csv_path = args.output_dir / "bvh_benchmark.csv"
    json_path = args.output_dir / "bvh_benchmark.json"
    fieldnames = [
        "mesh",
        "mesh_face_count",
        "backend",
        "available",
        "reason",
        "build_ms",
        "query_ms",
        "mae_vs_igl",
        "rmse_vs_igl",
        "max_abs_vs_igl",
        "point_mae_vs_igl",
        "point_rmse_vs_igl",
        "point_max_abs_vs_igl",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    leaderboard = build_leaderboard(rows)
    leaderboard_csv_path = args.output_dir / "bvh_leaderboard.csv"
    leaderboard_json_path = args.output_dir / "bvh_leaderboard.json"
    leaderboard_fields = [
        "backend",
        "num_meshes",
        "avg_build_ms",
        "rank_avg_build_ms",
        "avg_query_ms",
        "rank_avg_query_ms",
        "avg_mae_vs_igl",
        "avg_rmse_vs_igl",
        "rank_avg_rmse_vs_igl",
        "avg_max_abs_vs_igl",
        "avg_point_mae_vs_igl",
        "avg_point_rmse_vs_igl",
        "rank_avg_point_rmse_vs_igl",
        "avg_point_max_abs_vs_igl",
    ]
    with leaderboard_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=leaderboard_fields)
        writer.writeheader()
        writer.writerows(leaderboard)
    with leaderboard_json_path.open("w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)

    build_curve_path = args.output_dir / "build_time_vs_face_count.png"
    query_curve_path = args.output_dir / "query_time_vs_face_count.png"
    error_curve_path = args.output_dir / "nearest_point_rmse_vs_face_count.png"
    draw_metric_curve(
        rows,
        metric="build_ms",
        y_label="Build time (ms)",
        title="BVH Build Time vs Mesh Face Count",
        out_path=build_curve_path,
    )
    draw_metric_curve(
        rows,
        metric="query_ms",
        y_label="Query time (ms)",
        title="BVH Query Time vs Mesh Face Count",
        out_path=query_curve_path,
    )
    draw_metric_curve(
        rows,
        metric="point_rmse_vs_igl",
        y_label="Nearest-point RMSE vs igl ground truth",
        title="Nearest-Point Error (RMSE) vs Mesh Face Count",
        out_path=error_curve_path,
    )

    print(f"\nSaved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved build curve: {build_curve_path}")
    print(f"Saved query curve: {query_curve_path}")
    print(f"Saved error curve: {error_curve_path}")
    print(f"Saved leaderboard CSV: {leaderboard_csv_path}")
    print(f"Saved leaderboard JSON: {leaderboard_json_path}")

    print("\n=== Leaderboard (lower is better) ===")
    leaderboard_sorted = sorted(
        leaderboard,
        key=lambda r: (
            float("inf") if not np.isfinite(r["avg_point_rmse_vs_igl"]) else r["avg_point_rmse_vs_igl"],
            float("inf") if not np.isfinite(r["avg_query_ms"]) else r["avg_query_ms"],
            float("inf") if not np.isfinite(r["avg_build_ms"]) else r["avg_build_ms"],
        ),
    )
    for row in leaderboard_sorted:
        print(
            f"{row['backend']:12s} "
            f"build={row['avg_build_ms']:.3f}ms (rank {row['rank_avg_build_ms']})  "
            f"query={row['avg_query_ms']:.3f}ms (rank {row['rank_avg_query_ms']})  "
            f"point_rmse={row['avg_point_rmse_vs_igl']:.6e} (rank {row['rank_avg_point_rmse_vs_igl']})"
        )


if __name__ == "__main__":
    main()
