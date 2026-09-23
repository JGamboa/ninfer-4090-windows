"""Create the `--model` base directory for the `bonsai2_27b` recipe.

The Prism GGUF carries weights but not the HF config or tokenizer files the converter reads
from `--model`. Bonsai 2 27B is the Qwen3.8-27B architecture, so the base directory is
taken from the reference `.ninfer` whose MTP head the recipe copies: its text config (the
converter's normalized Qwen3.5 config, which `qwen3_5.text_config` accepts unchanged) and
its text resources (`tokenizer.json`, `tokenizer_config.json`, `generation_config.json`,
`chat_template.jinja`). The config is checked against the GGUF geometry, and the GGUF's own
`tokenizer.chat_template` is written next to the base as `gguf_chat_template.jinja` with a
line stating whether it matches the reference template (design doc section 8, risk 5).

    python -m tools.convert.bonsai_base --gguf E:\\LLM\\Ternary-Bonsai-2-27B-PTQ1_0.gguf ^
        --reference E:\\LLM\\qwen3_8_27b.ninfer --out E:\\LLM\\bonsai2-27b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .qwen3_5 import text_config
from .resources import TEXT_RESOURCES
from .sources.gguf_reader import GgufStore
from .sources.ninfer_artifact import NInferArtifactStore

_GGUF_FIELDS = {
    "hidden_size": "qwen35.embedding_length",
    "num_hidden_layers": "qwen35.block_count",
    "intermediate_size": "qwen35.feed_forward_length",
    "num_attention_heads": "qwen35.attention.head_count",
    "num_key_value_heads": "qwen35.attention.head_count_kv",
    "head_dim": "qwen35.attention.key_length",
    "linear_num_key_heads": "qwen35.ssm.group_count",
    "linear_key_head_dim": "qwen35.ssm.state_size",
    "linear_num_value_heads": "qwen35.ssm.time_step_rank",
    "linear_conv_kernel_dim": "qwen35.ssm.conv_kernel",
}


def write_base(gguf_path: Path, reference_path: Path, out: Path) -> dict:
    """Write config.json and text resources to `out`; return a small report."""
    with NInferArtifactStore(reference_path) as reference, GgufStore(gguf_path) as gguf:
        component = reference.directory.components["text"]
        config = dict(component["config"])
        config.pop("prism_hadamard", None)
        text_config(config, mtp=True)
        for key, meta_key in _GGUF_FIELDS.items():
            if config.get(key) != gguf.metadata.get(meta_key):
                raise ValueError(
                    f"{reference_path}: {key}={config.get(key)!r} differs from "
                    f"{gguf_path} {meta_key}={gguf.metadata.get(meta_key)!r}"
                )
        vocab = gguf.tensor("token_embd.weight").shape[1]
        if config["vocab_size"] != vocab:
            raise ValueError(f"vocab_size {config['vocab_size']} differs from GGUF {vocab}")
        resources = component.get("resources", {})
        missing = [role for role in TEXT_RESOURCES if role not in resources]
        if missing:
            raise ValueError(f"{reference_path}: missing text resources {missing}")
        out.mkdir(parents=True, exist_ok=False)
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        template = b""
        for role in TEXT_RESOURCES:
            data = reference.read_object(resources[role])
            (out / role).write_bytes(data)
            if role == "chat_template.jinja":
                template = data
        gguf_template = gguf.metadata.get("tokenizer.chat_template")
    report = {"config": str(out / "config.json"), "chat_template_matches_gguf": None}
    if isinstance(gguf_template, str):
        (out / "gguf_chat_template.jinja").write_text(gguf_template, encoding="utf-8")
        report["chat_template_matches_gguf"] = (
            gguf_template.strip() == template.decode("utf-8").strip()
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = write_base(args.gguf, args.reference, args.out)
    print(f"wrote {report['config']}")
    matches = report["chat_template_matches_gguf"]
    if matches is None:
        print("the GGUF has no tokenizer.chat_template")
    else:
        print(
            "chat template: reference "
            + ("matches" if matches else "DIFFERS FROM")
            + " the GGUF's (saved as gguf_chat_template.jinja)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
