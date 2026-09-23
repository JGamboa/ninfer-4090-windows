from __future__ import annotations

import torch

from tools.artifact.codecs.row_split import encode_row_split
from tools.artifact.schema import TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.quantization.groupwise import quantize_matrix
from tools.convert.sources.ninfer_artifact import NInferArtifactStore


def test_dequantize_direct_vector_and_row_split_matrix_bindings(tmp_path):
    norm = torch.arange(8, dtype=torch.float32).to(torch.bfloat16)
    weight = torch.randn(4, 64)
    quantized = quantize_matrix(weight, "q4_g64_fp16", device="cpu")
    payload = encode_row_split(quantized.codes, quantized.scales, "q4_g64_fp16", (4, 64))

    path = tmp_path / "tiny.ninfer"
    with ArtifactWriter(
        path,
        [
            TensorSpec("norm", (8,), "bf16", "contiguous_le_v1"),
            TensorSpec("matrix", (4, 64), "q4_g64_fp16", "row_split_k128_v1"),
        ],
        components={"text": {"config": {}}},
        bindings={
            "text/norm": {"object": "norm"},
            "text/layers/0/a": {"parts": [{"object": "matrix", "range": [0, 128]}]},
            "text/layers/0/b": {"parts": [{"object": "matrix", "range": [128, 256]}]},
            "text/layers/0/whole": {"object": "matrix"},
        },
    ) as writer:
        writer.write_object("norm", norm.contiguous().view(torch.uint8).numpy().tobytes())
        writer.write_object("matrix", payload)

    groups_per_row = quantized.codes.shape[1]
    expected = (
        quantized.codes.to(torch.float32) * quantized.scales.to(torch.float32).unsqueeze(-1)
    ).reshape(4, groups_per_row * 64)[:, :64]

    with NInferArtifactStore(path) as store:
        assert set(store.parameters()) == {
            "text/norm",
            "text/layers/0/a",
            "text/layers/0/b",
            "text/layers/0/whole",
        }
        got_norm = store.dequantize("text/norm")
        assert torch.equal(got_norm, norm.to(torch.float32))

        a = store.dequantize("text/layers/0/a")
        b = store.dequantize("text/layers/0/b")
        assert torch.equal(a, expected[0:2])
        assert torch.equal(b, expected[2:4])

        whole = store.dequantize("text/layers/0/whole")
        assert torch.equal(whole, expected)

    with NInferArtifactStore(path) as store:
        try:
            store.dequantize("text/does/not/exist")
        except KeyError:
            pass
        else:
            raise AssertionError("expected KeyError for an unbound parameter name")
