"""Official representation recipes built from the same public conversion functions."""

from __future__ import annotations

from dataclasses import replace

import torch

from tools.artifact.formats import DirectFormat, get_format
from .methods import cast_direct, fp8_row_maxabs, grouped_absmax, import_encoded
from .model import Parameter
from .sources.logical import array_source
from .sources.ninfer_artifact import NInferArtifactStore
from .sources.mmproj import MmprojCheckpoint
from .sources.prism_checkpoint import PrismCheckpoint

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"
FP8 = "fp8_e4m3fn_row_bf16"


def _assign(recipe, name, format, *, source=None):
    method = grouped_absmax if format in (Q4, Q5, Q6, Q8) else cast_direct
    recipe.assign(name, format=format, method=method, source=source)


def _optional(model, recipe):
    for name, parameter in model.parameters.items():
        if not parameter.projection:
            continue
        if name.startswith("vision/"):
            if name == "vision/patch_embedding":
                format = Q6
            elif name.startswith("vision/merger/"):
                format = Q8
            elif name.endswith(
                ("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")
            ):
                format = Q4
            else:
                format = Q5
            _assign(recipe, name, format)
        elif name.startswith(("mtp/", "dflash/", "dflash2/")):
            if name.endswith(
                (
                    "/moe/router",
                    "/moe/shared_score",
                    "/attention_conv/kernel_projection",
                    "/mlp_conv/kernel_projection",
                    "/candidate_selector/hidden_projection",
                )
            ):
                continue
            _assign(recipe, name, Q8)
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _dense_groupwise(model, recipe, vocabulary):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", vocabulary)
    _assign(recipe, "text/output_head", vocabulary)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        if name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/gdn/query",
                "/gdn/key",
                "/mlp/gate",
                "/mlp/up",
            )
        ):
            format = Q4
        else:
            format = Q5
        _assign(recipe, name, format)


def qwen3_6_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q6)


def qwen3_8_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8)


def qwen3_6_35b_a3b(model, recipe, sources):
    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name:
            layer = int(name.split("/")[2])
            format = (
                (Q6 if layer in (34, 38, 39) else Q5) if name.endswith("/down") else Q4
            )
        else:
            format = Q8
        _assign(recipe, name, format)


def qwen3_6_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        layer = int(name.split("/")[2])
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        direct = (
            ("/attention/" in name and not name.endswith("/output") and layer < 24)
            or (name.endswith("/attention/output") and layer in (3, 7))
            or (name.endswith("/gdn/output") and layer == 4)
        )
        if direct:
            continue
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=model.source(name, quantized, "nvfp4"),
            activation_policy="AllowA4",
        )


def qwen3_8_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        source = model.source(name, quantized)
        if not parameter.projection or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            recipe.assign(name, source=source)
            continue
        layer = int(name.split("/")[2]) if name.startswith("text/layers/") else -1
        format = "nvfp4" if "/mlp/" in name and layer < 56 else FP8
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


# The stored ternary format of every Bonsai projection, head and embedding. The base-3
# `t5_g128_fp16` replaces it once its production kernels land (design doc 9.1, step 2).
TERNARY_FORMAT = "t2_g128_fp16"

_BONSAI_TERNARY = (
    "/attention/query",
    "/attention/key",
    "/attention/gate",
    "/attention/value",
    "/attention/output",
    "/gdn/query",
    "/gdn/key",
    "/gdn/value",
    "/gdn/z",
    "/gdn/output",
    "/mlp/gate",
    "/mlp/up",
    "/mlp/down",
)
_MTP_CONFIG_FIELDS = (
    "hidden_size",
    "vocab_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "rms_norm_eps",
)


def _bonsai_geometry(model, gguf: PrismCheckpoint) -> None:
    config = model.config
    meta = gguf.metadata
    expected = {
        "hidden_size": meta.get("qwen35.embedding_length"),
        "num_hidden_layers": meta.get("qwen35.block_count"),
        "intermediate_size": meta.get("qwen35.feed_forward_length"),
        "num_attention_heads": meta.get("qwen35.attention.head_count"),
        "num_key_value_heads": meta.get("qwen35.attention.head_count_kv"),
        "head_dim": meta.get("qwen35.attention.key_length"),
        "linear_num_key_heads": meta.get("qwen35.ssm.group_count"),
        "linear_key_head_dim": meta.get("qwen35.ssm.state_size"),
        "linear_num_value_heads": meta.get("qwen35.ssm.time_step_rank"),
        "linear_conv_kernel_dim": meta.get("qwen35.ssm.conv_kernel"),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"bonsai: config {key}={config.get(key)!r}, GGUF says {value!r}")
    interval = meta.get("qwen35.full_attention_interval")
    layers = [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(config["num_hidden_layers"])
    ]
    if config["layer_types"] != layers:
        raise ValueError("bonsai: config layer_types differ from the GGUF attention interval")
    if config["tie_word_embeddings"]:
        raise ValueError("bonsai: the Prism GGUF has an untied output head")


def _bonsai_mtp(model, recipe, reference: NInferArtifactStore) -> None:
    """Copy MTP exactly: grouped-integer parents as stored words, direct values as stored."""
    component = reference.directory.components.get("text", {}).get("config", {})
    for key in _MTP_CONFIG_FIELDS:
        if component.get(key) != model.config.get(key):
            raise ValueError(f"bonsai: MTP reference {key} differs from the target config")
    for name, parameter in model.parameters.items():
        if not name.startswith("mtp/"):
            continue
        source = reference.parameter_source(name, parameter.shape)
        format = reference.stored_format(name)
        if isinstance(get_format(format), DirectFormat):
            recipe.assign(name, format=format, method=cast_direct, source=source)
        else:
            recipe.assign(name, format=format, method=import_encoded, source=source)


def bonsai2_27b(model, recipe, sources):
    """Prism Ternary Bonsai 2: t2 projections, output head and embedding, copied MTP.

    Sources: ``gguf`` (the PTQ1_0/PQ2_0 GGUF); with the ``mtp`` component, ``mtp`` (an
    existing Qwen3.8-27B ``.ninfer`` whose MTP head is copied word for word); with the
    ``vision`` component, ``mmproj`` (Prism's Qwen3-VL mmproj GGUF, not ternary), quantized
    to the official Vision formats (Q4/Q5/Q6/Q8, the registered Vision kernels).
    """
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    gguf = sources["gguf"]
    if not isinstance(gguf, PrismCheckpoint):
        raise ValueError("bonsai2_27b requires --source gguf=PATH.gguf")
    _bonsai_geometry(model, gguf)
    _optional(model, recipe)
    rotated = set()
    for name, parameter in list(model.parameters.items()):
        if not name.startswith("text/"):
            continue
        source = model.source(name, gguf)
        # The output head and --proposal read the primal-basis GGUF values.
        model.parameters[name] = replace(parameter, source=source)
        if name == "text/token_embedding":
            # The rotated ternary table; the gather applies signs * H z' / 32 per row.
            recipe.assign(
                name,
                format=TERNARY_FORMAT,
                method=import_encoded,
                source=model.source(name, gguf, TERNARY_FORMAT),
            )
        elif name == "text/output_head":
            # The rotated ternary head; the runtime rotates a copy of its input.
            recipe.assign(
                name,
                format=TERNARY_FORMAT,
                method=import_encoded,
                source=model.source(name, gguf, TERNARY_FORMAT),
                activation_policy="AllowA8",
            )
            rotated.add("output_head")
        elif name.startswith("text/layers/") and name.endswith(_BONSAI_TERNARY):
            recipe.assign(
                name,
                format=TERNARY_FORMAT,
                method=import_encoded,
                source=model.source(name, gguf, TERNARY_FORMAT),
                activation_policy="AllowA8",
            )
            rotated.add(name.split("/", 3)[3])
        else:
            if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
                recipe.separate(name)
            recipe.assign(name, source=source)
    signs = {}
    for width, values in sorted(gguf.signs.items()):
        name = f"text/hadamard/signs_{width}"
        model.add(
            Parameter(
                name,
                (width,),
                array_source(values.to(torch.bfloat16), f"{gguf.path}:signs[{width}]"),
            )
        )
        recipe.add_parameter(name)
        signs[str(width)] = name
    model.config["prism_hadamard"] = {
        "version": 1,
        "transform": "normalized-sylvester-walsh-hadamard",
        "block_size": 1024,
        "sign_mode": "explicit",
        "sign_widths": sorted(gguf.signs),
        "signs": signs,
        "rotated_inputs": sorted(rotated),
        "embedding_inverse": True,
    }
    if "mtp" in model.components:
        _bonsai_mtp(model, recipe, sources["mtp"])
    if "vision" in model.components:
        _bonsai_vision(model, recipe, sources["mmproj"])


def _bonsai_vision(model, recipe, mmproj):
    """Read every Vision parameter from the mmproj; formats stay those of `_optional`."""
    if not isinstance(mmproj, MmprojCheckpoint):
        raise ValueError("bonsai2_27b vision requires --source mmproj=PATH-mmproj.gguf")
    for name, parameter in list(model.parameters.items()):
        if name.startswith("vision/"):
            source = model.source(name, mmproj)
            model.parameters[name] = replace(parameter, source=source)
            recipe.assign(name, source=source)


RECIPES = {
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
    "bonsai2_27b": bonsai2_27b,
}
