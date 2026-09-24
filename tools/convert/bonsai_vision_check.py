"""Compare a Prism Bonsai vision mmproj with the Vision tower of a Qwen3.8 `.ninfer`.

Answers whether Prism kept the base Vision weights: for a sample of Vision parameters it
prints the relative L2 difference between the reference artifact's (dequantized) values
and the mmproj's values mapped to the same HF layout. Identical towers differ only by the
two quantizations (about 0.01 for Q8 up to about 0.1 for Q4); different weights give
differences near 1 or above.

    python -m tools.convert.bonsai_vision_check ^
        --mmproj E:\\LLM\\bonsai\\Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf ^
        --reference E:\\LLM\\qwen3_8_27b.ninfer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .sources.mmproj import MmprojCheckpoint
from .sources.ninfer_artifact import NInferArtifactStore


def _samples(depth: int, h: int) -> list[tuple[str, str, slice | None]]:
    """(NInfer parameter, mmproj HF name, row slice of the HF tensor)."""
    rows = []
    for i in sorted({0, depth // 2, depth - 1}):
        p, s = f"vision/layers/{i}/", f"model.visual.blocks.{i}."
        rows += [
            (p + "attention/query", s + "attn.qkv.weight", slice(0, h)),
            (p + "attention/value", s + "attn.qkv.weight", slice(2 * h, 3 * h)),
            (p + "attention/output", s + "attn.proj.weight", None),
            (p + "mlp/fc1", s + "mlp.linear_fc1.weight", None),
            (p + "mlp/fc2", s + "mlp.linear_fc2.weight", None),
            (p + "norm1_weight", s + "norm1.weight", None),
        ]
    rows += [
        ("vision/patch_embedding", "model.visual.patch_embed.proj.weight", None),
        ("vision/position_embedding", "model.visual.pos_embed.weight", None),
        ("vision/merger/fc1", "model.visual.merger.linear_fc1.weight", None),
        ("vision/merger/fc2", "model.visual.merger.linear_fc2.weight", None),
    ]
    return rows


def compare(mmproj_path: Path, reference_path: Path) -> list[tuple[str, float]]:
    results = []
    with MmprojCheckpoint(mmproj_path) as mmproj, NInferArtifactStore(reference_path) as ref:
        if "vision" not in ref.directory.components:
            raise ValueError(f"{reference_path} has no vision component")
        config = mmproj.config["vision_config"]
        for name, hf, rows in _samples(config["depth"], config["hidden_size"]):
            values = mmproj.values(hf)
            values = values[rows] if rows is not None else values
            reference = ref.dequantize(name).to(torch.float64).reshape(-1)
            values = values.to(torch.float64).reshape(-1)
            if reference.numel() != values.numel():
                raise ValueError(f"{name}: {reference.numel()} vs {values.numel()} values")
            results.append(
                (name, float(torch.linalg.norm(values - reference) / torch.linalg.norm(reference)))
            )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args(argv)
    results = compare(args.mmproj, args.reference)
    for name, difference in results:
        print(f"{name:40s} relative L2 {difference:.4f}")
    worst = max(difference for _, difference in results)
    print(
        "verdict: "
        + ("same Vision weights (within quantization)" if worst < 0.2 else "DIFFERENT weights")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
