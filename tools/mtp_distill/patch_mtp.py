"""Copy a .ninfer artifact and rewrite only its MTP objects with trained values.

  python -m tools.mtp_distill.patch_mtp SRC.ninfer DST.ninfer WEIGHTS.pt
  python -m tools.mtp_distill.patch_mtp SRC.ninfer - --check

Every MTP parameter keeps its stored format, shape and object; grouped-integer objects are
requantized with the converter's own `quantize_matrix` + `encode_row_split`, BF16 values are
written as BF16. Only the bytes of objects that hold MTP parameters change. `--check` only
verifies that re-encoding each original MTP object from its decoded values reproduces it.
"""

from __future__ import annotations

import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch

from tools.artifact.codecs.row_split import dequantize_row_split, encode_row_split
from tools.artifact.formats import DirectFormat, QuantFormat, get_format
from tools.convert.quantization.groupwise import quantize_matrix
from tools.convert.sources.ninfer_artifact import NInferArtifactStore


def mtp_objects(store: NInferArtifactStore):
    """object_id -> [(name, begin, end)] for every MTP binding."""
    out = defaultdict(list)
    for name in store.parameters():
        if not name.startswith("mtp/"):
            continue
        parts, objects = store._parts(name)
        for object_id, begin, end in parts:
            out[object_id].append((name, begin, end))
    return out, objects


def encode(values: torch.Tensor, obj) -> bytes:
    fmt = get_format(obj.format)
    q = quantize_matrix(values, fmt, device="cuda")
    return encode_row_split(q.codes.cpu(), q.scales.cpu(), fmt, obj.shape)


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    check = "--check" in sys.argv
    trained = {} if check else torch.load(sys.argv[3])
    patches: list[tuple[int, bytes]] = []
    with NInferArtifactStore(src) as store:
        art = store._artifact
        groups, objects = mtp_objects(store)
        names = {n for entries in groups.values() for n, _, _ in entries}
        if not check and names != set(trained):
            raise SystemExit(f"weights do not match the MTP bindings: {sorted(names ^ set(trained))}")
        for object_id, entries in sorted(groups.items()):
            obj = objects[object_id]
            fmt = get_format(obj.format)
            original = art.read_object(object_id)
            base = art.payload_offset + obj.offset
            if isinstance(fmt, QuantFormat):
                n, k = obj.shape
                values = dequantize_row_split(original, fmt, (n, k), dtype=torch.float32)
                if check:
                    same = encode(values, obj) == original
                    print(f"{object_id} {obj.format} {list(obj.shape)} re-encode identical: {same}")
                    continue
                for name, begin, end in entries:
                    rows = trained[name].float().reshape(-1, k)
                    if rows.shape[0] != (end - begin) // k:
                        raise SystemExit(f"{name}: {tuple(rows.shape)} does not fit [{begin},{end}) of {obj.shape}")
                    values[begin // k : end // k] = rows
                payload = encode(values, obj)
                if len(payload) != len(original):
                    raise SystemExit(f"{object_id}: payload size changed")
                patches.append((base, payload))
                print(f"{object_id} {obj.format} {list(obj.shape)} <- {[e[0] for e in entries]}")
            elif isinstance(fmt, DirectFormat):
                if check:
                    continue
                for name, begin, end in entries:
                    if fmt.name != "bf16":
                        raise SystemExit(f"{name}: unexpected direct format {fmt.name}")
                    data = trained[name].to(torch.bfloat16).reshape(-1)
                    if data.numel() != end - begin:
                        raise SystemExit(f"{name}: size mismatch")
                    patches.append((base + begin * 2, data.view(torch.uint16).numpy().tobytes()))
                print(f"{object_id} {obj.format} {list(obj.shape)} <- {[e[0] for e in entries]}")
            else:
                raise SystemExit(f"{object_id}: unsupported format {obj.format}")
    if check:
        return
    shutil.copyfile(src, dst)
    with open(dst, "r+b") as f:
        for offset, payload in patches:
            f.seek(offset)
            f.write(payload)
    print(f"wrote {dst}: {len(patches)} patches")


if __name__ == "__main__":
    main()
