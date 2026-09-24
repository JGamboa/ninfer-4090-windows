"""Compare every Bonsai logical parameter of selected layers with the Qwen3.8-27B artifact.

M0 (`bonsai_m0_check`) proved the conventions on a few tensors. This check runs the whole
`bonsai2_27b` mapping (`PrismCheckpoint` through the Qwen3.5 builder) for one GDN layer, one
attention layer and the global tensors, and compares the primal-basis values per row with
the same logical parameter dequantized from the reference artifact. Ternary tensors are
expected near the M0 plateau (median row cosine ~0.88), direct tensors near 1; a wrong head
order, norm offset or q/gate split collapses to ~0.

    python -m tools.convert.bonsai_mapping_check --gguf E:\\LLM\\Ternary-Bonsai-2-27B-PTQ1_0.gguf ^
        --reference E:\\LLM\\qwen3_8_27b.ninfer --base E:\\LLM\\bonsai2-27b
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .qwen3_5 import build_model
from .sources.ninfer_artifact import NInferArtifactStore
from .sources.prism_checkpoint import PrismCheckpoint
from .sources.safetensors import SafetensorsSource

TERNARY_MIN, DIRECT_MIN = 0.80, 0.98


def _row_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.double(), b.double()
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-30)


def compare(gguf: Path, reference: Path, base: Path, layers=(0, 3), max_rows=2048):
    rows_out = []
    with (
        PrismCheckpoint(gguf) as checkpoint,
        NInferArtifactStore(reference) as ref,
        SafetensorsSource(base) as store,
    ):
        model = build_model(store)
        prefixes = tuple(f"text/layers/{i}/" for i in layers)
        names = [
            n
            for n in model.parameters
            if n.startswith(prefixes) or n in ("text/final_norm", "text/output_head", "text/token_embedding")
        ]
        for name in names:
            parameter = model.parameters[name]
            source = model.source(name, checkpoint)
            if len(parameter.shape) == 2:
                count = min(parameter.shape[0], max_rows)
                ours = source.rows(0, count)
                theirs = ref.dequantize(name).reshape(parameter.shape)[:count]
                cosine = _row_cosine(ours, theirs)
                ternary = source.read_encoded is not None
                value = float(cosine.median())
                extra = f"min {float(cosine.min()):.4f}"
            else:
                ours = source.values().reshape(1, -1)
                theirs = ref.dequantize(name).reshape(1, -1)
                ternary = False
                value = float(_row_cosine(ours, theirs)[0])
                extra = f"max|diff| {float((ours - theirs).abs().max()):.4g}"
            threshold = TERNARY_MIN if ternary else DIRECT_MIN
            rows_out.append((name, "ternary" if ternary else "direct", value, extra, value >= threshold))
    return rows_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True, help="directory from bonsai_base")
    parser.add_argument("--layers", default="0,3")
    parser.add_argument("--max-rows", type=int, default=2048)
    args = parser.parse_args(argv)
    layers = tuple(int(i) for i in args.layers.split(","))
    results = compare(args.gguf, args.reference, args.base, layers, args.max_rows)
    print("| parameter | kind | cosine | detail | ok |\n|---|---|---|---|---|")
    for name, kind, value, extra, ok in results:
        print(f"| `{name}` | {kind} | {value:.4f} | {extra} | {'yes' if ok else 'NO'} |")
    return 0 if all(ok for *_, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
