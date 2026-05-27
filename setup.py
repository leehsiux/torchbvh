from pathlib import Path

from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent.resolve()

eigen_include = Path("/usr/include/eigen3")
include_dirs = [str(ROOT / "torchbvh" / "src")]
if eigen_include.exists():
    include_dirs.append(str(eigen_include))
torch_include = Path(torch.__file__).resolve().parent / "include"
torch_eigen = torch_include / "third_party" / "eigen3"
if torch_eigen.exists():
    include_dirs.append(str(torch_eigen))

ext_modules = [
    CUDAExtension(
        name="torchbvh._C",
        sources=[
            str(ROOT / "torchbvh" / "src" / "bvh" / "bvh.cpp"),
            str(ROOT / "torchbvh" / "src" / "bvh" / "bindings.cpp"),
            str(ROOT / "torchbvh" / "src" / "bvh" / "bvh_device.cu"),
        ],
        include_dirs=include_dirs,
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            # Avoid fast-math to preserve query precision fidelity.
            "nvcc": ["-O3", "-lineinfo"],
        },
    )
]

setup(
    name="torchbvh",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
