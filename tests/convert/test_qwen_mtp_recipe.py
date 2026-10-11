"""`qwen3_{8,6}_27b_a8_mtp_q4q5` from a reference artifact of the matching non-A8 recipe.

The reference is a random Qwen3.8-shaped artifact (the Bonsai fixture's small geometry). The
oracle is the reference's own stored words and decoded values: every parameter outside the MTP
layer's projections must be copied word for word, and each MTP layer projection must equal a
symmetric grouped absmax of the reference's decoded values in the official Q4/Q5 mix.
"""

from __future__ import annotations

import pytest
import torch

from tools.artifact.formats import DirectFormat, get_format
from tools.convert.__main__ import main as convert_main
from tools.convert.official_recipes import qwen3_6_27b, qwen3_8_27b
from tools.convert.pipeline import convert
from tools.convert.qwen3_5 import build_model
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source
from tools.convert.sources.ninfer_artifact import NInferArtifactStore
from tools.convert.sources.safetensors import SafetensorsSource

from .bonsai_fixtures import write_resources

_LAYER = "mtp/layers/0/"
_ROLES = (
    "attention/query", "attention/key", "attention/gate", "attention/value", "attention/output",
    "mlp/gate", "mlp/up", "mlp/down",
)
_Q4 = {"attention/query", "attention/key", "mlp/gate", "mlp/up"}
_QMAX = {"q4_g64_fp16": 7, "q5_g64_fp16": 15}


_CASES = {"qwen3_8_27b_a8_mtp_q4q5": qwen3_8_27b, "qwen3_6_27b_a8_mtp_q4q5": qwen3_6_27b}


@pytest.fixture(scope="module", params=sorted(_CASES))
def converted(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp(request.param)
    config_dir = tmp / "config"
    write_resources(config_dir)
    reference = tmp / "reference.ninfer"
    with SafetensorsSource(config_dir) as store:
        model = build_model(store, components=("text", "mtp"))
        recipe = Recipe(model)
        generator = torch.Generator().manual_seed(5)
        for name, parameter in model.parameters.items():
            recipe.assign(name, source=array_source(torch.randn(parameter.shape, generator=generator), name))
        _CASES[request.param](model, recipe, {})
        convert(model, recipe, reference, device="cpu")
    out = tmp / "mtp_q4q5.ninfer"
    convert_main(
        [
            "--model", str(config_dir),
            "--recipe", request.param,
            "--components", "text,mtp",
            "--source", f"reference={reference}",
            "--device", "cpu",
            "--out", str(out),
        ]
    )
    return reference, out


def test_mtp_layer_is_requantized_to_the_official_mix(converted):
    reference, out = converted
    with NInferArtifactStore(reference) as source, NInferArtifactStore(out) as result:
        for role in _ROLES:
            name = _LAYER + role
            format = "q4_g64_fp16" if role in _Q4 else "q5_g64_fp16"
            assert source.stored_format(name) == "q8_g32_fp16", name
            assert result.stored_format(name) == format, name
            values = source.dequantize(name).double()
            groups = values.reshape(values.shape[0], -1, 64)
            step = groups.abs().amax(dim=2, keepdim=True) / _QMAX[format]
            error = (result.dequantize(name).double().reshape(groups.shape) - groups).abs()
            assert bool((error <= step * (0.5 + 2.0**-9)).all()), name
            assert float(error.max()) > 0, name


def test_everything_else_is_copied_word_for_word(converted):
    reference, out = converted
    requantized = {_LAYER + role for role in _ROLES}
    with NInferArtifactStore(reference) as source, NInferArtifactStore(out) as result:
        names = set(source.parameters())
        assert names == set(result.parameters())
        for name in sorted(names - requantized):
            format = source.stored_format(name)
            assert result.stored_format(name) == format, name
            if isinstance(get_format(format), DirectFormat):
                assert torch.equal(source.dequantize(name), result.dequantize(name)), name
            else:
                rows = source.dequantize(name).shape[0]
                a, b = source.encoded_rows(name, 0, rows), result.encoded_rows(name, 0, rows)
                assert torch.equal(a.codes, b.codes) and torch.equal(a.scales, b.scales), name
