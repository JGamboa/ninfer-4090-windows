"""Read-only, bindings-addressed access to an existing .ninfer artifact.

Used to copy the Qwen3.8-27B MTP component into the Bonsai artifact unchanged, and by the
M0 acceptance test as the ground truth that the ternary decode + Hadamard reconstruction is
compared against. :meth:`NInferArtifactStore.parameter_source` exposes a bound parameter as
a logical source: float32 values, and for grouped-integer parents the exact stored codes and
scales read with bounded plane ranges (`import_encoded` pass-through).
"""

from __future__ import annotations

from pathlib import Path

import torch

from tools.artifact.formats import DirectFormat, QuantFormat, get_format
from tools.artifact.codecs.row_split import (
    RowPlanes,
    decode_row_split_codes,
    dequantize_row_split,
)
from tools.artifact.layouts import row_split_geometry
from tools.convert.sources.logical import EncodedRows, LogicalSource
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

    def read_object(self, object_id: str) -> bytes:
        return self._artifact.read_object(object_id)

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

    def stored_format(self, parameter_name: str) -> str:
        """The single numeric format of every part bound to `parameter_name`."""
        parts, objects = self._parts(parameter_name)
        formats = {objects[object_id].format for object_id, _, _ in parts}
        if len(formats) != 1:
            raise ArtifactError(f"{parameter_name}: parts use several formats {sorted(formats)}")
        return formats.pop()

    def _row_parts(self, parameter_name: str):
        """`(object, first_row, last_row)` spans of a matrix parameter bound to Q parents."""
        parts, objects = self._parts(parameter_name)
        spans = []
        for object_id, begin, end in parts:
            obj = objects[object_id]
            if not isinstance(get_format(obj.format), QuantFormat) or len(obj.shape) != 2:
                raise ArtifactError(f"{parameter_name}: {object_id} is not a grouped matrix")
            k = obj.shape[1]
            if begin % k or end % k:
                raise ArtifactError(f"{parameter_name}: {object_id} part is not whole rows")
            spans.append((obj, begin // k, end // k))
        return spans

    def encoded_rows(self, parameter_name: str, begin: int, end: int) -> EncodedRows:
        """Exact grouped codes `[rows, groups, group]` and FP16 scales for logical rows."""
        pieces = []
        cursor = 0
        for obj, first, last in self._row_parts(parameter_name):
            low, high = max(begin, cursor), min(end, cursor + last - first)
            if low < high:
                fmt = get_format(obj.format)
                geometry = row_split_geometry(fmt, obj.shape)
                row, count = first + low - cursor, high - low

                def plane(offset, row_bytes):
                    return self._artifact.read_range(
                        obj.offset + offset + row * row_bytes, count * row_bytes
                    )

                planes = RowPlanes(
                    plane(geometry.base_offset, geometry.base_row_bytes),
                    plane(geometry.high_offset, geometry.high_row_bytes),
                    plane(geometry.scale_offset, geometry.scale_row_bytes),
                    count,
                )
                scales, codes = decode_row_split_codes(planes, fmt, (count, obj.shape[1]))
                pieces.append((obj.format, codes, scales))
            cursor += last - first
        if not pieces or not 0 <= begin < end <= cursor:
            raise ArtifactError(f"{parameter_name}: invalid encoded rows [{begin},{end})")
        if len({format for format, _, _ in pieces}) != 1:
            raise ArtifactError(f"{parameter_name}: encoded parts use several formats")
        return EncodedRows(
            pieces[0][0],
            torch.cat([codes for _, codes, _ in pieces]),
            torch.cat([scales for _, _, scales in pieces]),
        )

    def parameter_source(self, parameter_name: str, shape: tuple[int, ...]) -> LogicalSource:
        """Logical source for a bound parameter; values are materialized once on demand."""
        cache: list[torch.Tensor] = []

        def values(begin: int, end: int) -> torch.Tensor:
            if not cache:
                cache.append(self.dequantize(parameter_name).reshape(-1))
            return cache[0][begin:end]

        encoded = None
        if isinstance(get_format(self.stored_format(parameter_name)), QuantFormat):
            def encoded(begin: int, end: int) -> EncodedRows:
                return self.encoded_rows(parameter_name, begin, end)

        return LogicalSource(tuple(shape), f"{self.path}:{parameter_name}", values, encoded)


__all__ = ["NInferArtifactStore"]
