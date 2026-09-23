"""Read-only, bindings-addressed access to an existing .ninfer artifact.

Used to copy the Qwen3.8-27B MTP/DFlash2 components into the Bonsai artifact unchanged,
and by the M0 acceptance test as the ground truth that the ternary decode + Hadamard
reconstruction is compared against. Values come back dequantized to float32; this module
never repacks encoded rows (`import_encoded`'s exact-bytes path is a separate,
not-yet-implemented step, design doc section 4).
"""

from __future__ import annotations

from pathlib import Path

import torch

from tools.artifact.formats import DirectFormat, QuantFormat, get_format
from tools.artifact.codecs.row_split import dequantize_row_split
from tools.artifact.reader import Artifact
from tools.artifact.schema import ArtifactError, binding_parts

_DIRECT_DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "int32": torch.int32,
}


class NInferArtifactStore:
    """Dequantize logical parameters of one `.ninfer` artifact by their bound name."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._artifact = Artifact.open(self.path)

    def close(self) -> None:
        self._artifact.close()

    def __enter__(self) -> "NInferArtifactStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def directory(self):
        return self._artifact.directory

    def parameters(self) -> tuple[str, ...]:
        """Every bound logical parameter name (e.g. `text/layers/0/gdn/output`)."""
        return tuple(self.directory.bindings)

    def _parts(self, parameter_name: str):
        objects = {obj.id: obj for obj in self.directory.objects}
        try:
            binding = self.directory.bindings[parameter_name]
        except KeyError as error:
            raise KeyError(
                f"{self.path}: no such bound parameter {parameter_name!r}"
            ) from error
        return binding_parts(binding, objects, parameter_name), objects

    def dequantize(
        self, parameter_name: str, *, device: str | torch.device = "cpu"
    ) -> torch.Tensor:
        """Return the logical values bound to `parameter_name` as float32.

        The result is `[rows, K]` when every part is a whole-row span of a matrix object
        of consistent K (the common case for a fused-parent projection slice), otherwise a
        flat `[elements]` vector (norms, biases, and other 1-D parameters).
        """
        parts, objects = self._parts(parameter_name)
        pieces: list[torch.Tensor] = []
        matrix_k: int | None = None
        for object_id, begin, end in parts:
            obj = objects[object_id]
            fmt = get_format(obj.format)
            if isinstance(fmt, DirectFormat):
                payload = self._artifact.read_range(
                    obj.offset + begin * fmt.word_bytes,
                    (end - begin) * fmt.word_bytes,
                )
                dtype = _DIRECT_DTYPES[fmt.name]
                values = torch.frombuffer(bytearray(payload), dtype=dtype)
                pieces.append(values.to(torch.float32))
            elif isinstance(fmt, QuantFormat):
                if len(obj.shape) != 2:
                    raise ArtifactError(
                        f"{parameter_name}: quantized part {object_id} is not a matrix"
                    )
                n, k = obj.shape
                if begin % k or end % k:
                    raise ArtifactError(
                        f"{parameter_name}: part {object_id} range [{begin},{end}) is "
                        f"not a whole-row span of K={k}"
                    )
                if matrix_k is None:
                    matrix_k = k
                elif matrix_k != k:
                    raise ArtifactError(
                        f"{parameter_name}: parts disagree on K ({matrix_k} vs {k})"
                    )
                row_begin, row_end = begin // k, end // k
                payload = self._artifact.read_object(object_id)
                full = dequantize_row_split(payload, fmt, (n, k), dtype=torch.float32)
                pieces.append(full[row_begin:row_end].reshape(-1))
            else:
                raise ArtifactError(
                    f"{parameter_name}: unsupported format {fmt.name} for dequantize"
                )
        flat = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
        if matrix_k is not None and flat.numel() % matrix_k == 0:
            return flat.reshape(-1, matrix_k).to(device)
        return flat.to(device)


__all__ = ["NInferArtifactStore"]
