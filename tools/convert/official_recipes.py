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


# The official dense Q4/Q5 mix: Q4 for the query/key and gate/up banks, Q5 for the rest.
_MIX_Q4 = (
    "/attention/query",
    "/attention/key",
    "/gdn/query",
    "/gdn/key",
    "/mlp/gate",
    "/mlp/up",
)


def _mixed_format(name):
    return Q4 if name.endswith(_MIX_Q4) else Q5


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
        _assign(recipe, name, _mixed_format(name))


def qwen3_6_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q6)


def qwen3_8_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8)


# The inputs of the dense text layers' Q4/Q5 GEMMs. With AllowA8 their prefill widths may run the
# int8 tensor-core GEMMs (per-token 64-group activation quantization); decode stays A16.
_PREFILL_A8 = (
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


def _import_stored(model, recipe, reference: NInferArtifactStore) -> None:
    """Take every parameter, the indexed proposal head and the resources from `reference`, an
    artifact of the same recipe: grouped-integer parents as stored codes and scales
    (`import_encoded`), direct values as stored. Formats, parents and sharing stay those the
    recipe already selected; a format that differs from the stored one fails."""
    stored = set(reference.parameters())
    for name, parameter in list(model.parameters.items()):
        if name in recipe.aliases:
            continue
        if name not in stored:
            raise ValueError(f"reference artifact has no parameter {name}")
        format = reference.stored_format(name)
        if any(selection.format != format for selection in recipe.selections[name]):
            raise ValueError(f"{name}: reference stores {format}, the recipe selects another format")
        source = reference.parameter_source(name, parameter.shape)
        model.parameters[name] = replace(parameter, source=source)
        method = cast_direct if isinstance(get_format(format), DirectFormat) else import_encoded
        recipe.assign(name, method=method, source=source)
    proposal = reference.directory.components.get("text", {}).get("proposal")
    if proposal is not None:
        rows = proposal["rows"]
        inputs = tuple(
            component + "/final_hidden"
            for component in ("mtp", "dflash", "dflash2")
            if component in model.components
        )
        head = reference.parameter_source("proposal/head", (rows, model.config["hidden_size"]))
        ids = reference.parameter_source("proposal/token_ids", (rows,))
        model.components["text"]["proposal"] = dict(proposal)
        model.add(Parameter("proposal/head", head.shape, head, inputs=inputs, residency="proposal"))
        model.add(
            Parameter(
                "proposal/token_ids", (rows,), ids, direct_format="int32", residency="proposal"
            )
        )
        recipe.add_parameter("proposal/head")
        recipe.add_parameter("proposal/token_ids")
        recipe.assign(
            "proposal/head",
            format=reference.stored_format("proposal/head"),
            method=import_encoded,
        )
    for object_id in model.resources:
        model.resources[object_id] = reference.read_object(object_id)


def qwen3_8_27b_a8(model, recipe, sources):
    """`qwen3_8_27b` with AllowA8 on the inputs of the text layers' Q4/Q5 GEMMs.

    With a `reference` source (an existing `qwen3_8_27b` artifact) the weights, the indexed
    proposal head and the resources are copied from its stored words instead of quantized from
    the BF16 checkpoint; `--model` then supplies only the configuration. Do not pass
    `--proposal` in that case."""
    _dense_groupwise(model, recipe, Q8)
    if "reference" in sources:
        _import_stored(model, recipe, sources["reference"])
    for name, parameter in model.parameters.items():
        if name.startswith("text/layers/") and name.endswith(_PREFILL_A8):
            recipe.assign(name, activation_policy="AllowA8")


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


# The stored ternary format of every Bonsai projection, head and embedding: scaled base 3,
# the exact GGUF trits (design doc section 2 and 9.1).
TERNARY_FORMAT = "t5_g128_fp16"

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


# Formats of the MTP layer's projections (attention q/k/gate/v/output, MLP gate/up/down) for
# the `bonsai2_27b_mtp_*` recipes, which requantize them from the reference's decoded values
# to cut the draft cost (design doc section 9.1). `mtp/input_projection` ([5120,10240]) keeps
# the copied Q8 words in every variant.
BONSAI_MTP_LAYER_FORMATS = {
    "q5": lambda name: Q5,
    "q4": lambda name: Q4,
    "q4q5": _mixed_format,
}


def _bonsai_mtp(model, recipe, reference: NInferArtifactStore, layer_format=None) -> None:
    """Copy MTP exactly: grouped-integer parents as stored words, direct values as stored.

    With `layer_format` (name -> format), the projections under `mtp/layers/` are instead
    quantized with grouped absmax from the reference's decoded values."""
    component = reference.directory.components.get("text", {}).get("config", {})
    for key in _MTP_CONFIG_FIELDS:
        if component.get(key) != model.config.get(key):
            raise ValueError(f"bonsai: MTP reference {key} differs from the target config")
    for name, parameter in model.parameters.items():
        if not name.startswith("mtp/"):
            continue
        source = reference.parameter_source(name, parameter.shape)
        format = reference.stored_format(name)
        if layer_format is not None and parameter.projection and name.startswith("mtp/layers/"):
            _assign(recipe, name, layer_format(name), source=source)
        elif isinstance(get_format(format), DirectFormat):
            recipe.assign(name, format=format, method=cast_direct, source=source)
        else:
            recipe.assign(name, format=format, method=import_encoded, source=source)


def bonsai2_27b(model, recipe, sources, mtp_layer=None):
    """Prism Ternary Bonsai 2: t5 projections, output head and embedding, copied MTP.

    Sources: ``gguf`` (the PTQ1_0/PQ2_0 GGUF); with the ``mtp`` component, ``mtp`` (an
    existing Qwen3.8-27B ``.ninfer`` whose MTP head is copied word for word, or with its layer
    requantized to the `BONSAI_MTP_LAYER_FORMATS[mtp_layer]` formats); with the
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
        layer_format = None if mtp_layer is None else BONSAI_MTP_LAYER_FORMATS[mtp_layer]
        _bonsai_mtp(model, recipe, sources["mtp"], layer_format)
    elif mtp_layer is not None:
        raise ValueError(f"bonsai2_27b_mtp_{mtp_layer} requires the mtp component")
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


def _bonsai2_27b_mtp(layer):
    def configure(model, recipe, sources):
        bonsai2_27b(model, recipe, sources, mtp_layer=layer)

    configure.__name__ = f"bonsai2_27b_mtp_{layer}"
    return configure


RECIPES = {
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_a8": qwen3_8_27b_a8,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
    "bonsai2_27b": bonsai2_27b,
    **{f"bonsai2_27b_mtp_{layer}": _bonsai2_27b_mtp(layer) for layer in BONSAI_MTP_LAYER_FORMATS},
}
