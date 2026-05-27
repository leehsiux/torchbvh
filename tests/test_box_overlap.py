import torch

import torchbvh


def test_box_overlap_query_returns_csr_face_candidates():
    if not torch.cuda.is_available():
        return

    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32, device="cuda")
    bvh = torchbvh.build_bvh(vertices, faces)

    lower = torch.tensor(
        [
            [-1e-4, -1e-4, -1e-4],
            [0.5 - 1e-4, 0.5 - 1e-4, -1e-4],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    upper = torch.tensor(
        [
            [1e-4, 1e-4, 1e-4],
            [0.5 + 1e-4, 0.5 + 1e-4, 1e-4],
        ],
        dtype=torch.float32,
        device="cuda",
    )

    offsets, indices = torchbvh.box_overlap_query(lower, upper, bvh)
    offsets = offsets.cpu()
    indices = indices.cpu()

    assert offsets.shape == (3,)
    assert offsets[0].item() == 0
    assert offsets[-1].item() == indices.numel()
    assert set(indices[offsets[0] : offsets[1]].tolist()) == {0, 1}
    assert set(indices[offsets[1] : offsets[2]].tolist()) == {0, 1}
