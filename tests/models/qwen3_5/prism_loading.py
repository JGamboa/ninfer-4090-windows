"""Write synthetic Bonsai artifacts with the production converter and plan them in C++."""

from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

try:
    import numpy  # noqa: F401
    import torch  # noqa: F401
except ImportError:
    raise SystemExit(77)

from tests.convert.bonsai_fixtures import convert_bonsai  # noqa: E402


def without_prism_block(model, recipe, sources):
    del model.config["prism_hadamard"]


def without_down_rotation(model, recipe, sources):
    model.config["prism_hadamard"]["rotated_inputs"].remove("mlp/down")


def down_not_ternary(model, recipe, sources):
    recipe.assign("text/layers/*/mlp/down", format="q8_g32_fp16", method="grouped_absmax")


def t5_a16_use(model, recipe, sources):
    recipe.assign("text/layers/*/mlp/down", activation_policy="A16Only")


def embedding_not_ternary(model, recipe, sources):
    recipe.assign("text/token_embedding", format="q8_g32_fp16", method="grouped_absmax")


CASES = (
    ("without_prism_block", "is not a prism_hadamard rotated weight"),
    ("without_down_rotation", "mlp/down: ternary weight is not a prism_hadamard"),
    ("down_not_ternary", "lists this weight but it is not ternary"),
    ("embedding_not_ternary", "token_embedding: prism_hadamard lists this weight"),
    ("t5_a16_use", "t5_g128_fp16 requires an AllowA8 use"),
)


def main() -> int:
    executable = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="ninfer-prism-loading-") as temporary:
        root = Path(temporary)
        (root / "valid").mkdir()
        *_, out, _ = convert_bonsai(root / "valid")
        if subprocess.run([executable, str(out)]).returncode:
            return 1
        for name, message in CASES:
            directory = root / name
            directory.mkdir()
            *_, out, _ = convert_bonsai(directory, "--override", f"{__file__}:{name}")
            if subprocess.run([executable, "--reject", message, str(out)]).returncode:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
