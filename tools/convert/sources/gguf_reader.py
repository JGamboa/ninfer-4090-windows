#!/usr/bin/env python3
"""Dependency-free GGUF v3 inventory: metadata keys + tensor table."""
import struct
import sys
from collections import Counter

GGUF_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
}

class R:
    def __init__(self, f):
        self.f = f
    def u8(self): return struct.unpack("<B", self.f.read(1))[0]
    def i8(self): return struct.unpack("<b", self.f.read(1))[0]
    def u16(self): return struct.unpack("<H", self.f.read(2))[0]
    def i16(self): return struct.unpack("<h", self.f.read(2))[0]
    def u32(self): return struct.unpack("<I", self.f.read(4))[0]
    def i32(self): return struct.unpack("<i", self.f.read(4))[0]
    def u64(self): return struct.unpack("<Q", self.f.read(8))[0]
    def i64(self): return struct.unpack("<q", self.f.read(8))[0]
    def f32(self): return struct.unpack("<f", self.f.read(4))[0]
    def f64(self): return struct.unpack("<d", self.f.read(8))[0]
    def boolean(self): return self.u8() != 0
    def string(self):
        n = self.u64()
        return self.f.read(n).decode("utf-8", errors="replace")
    def value(self, t):
        return {
            0: self.u8, 1: self.i8, 2: self.u16, 3: self.i16, 4: self.u32, 5: self.i32,
            6: self.f32, 7: self.boolean, 8: self.string, 10: self.u64, 11: self.i64, 12: self.f64,
        }[t]() if t != 9 else self.array()
    def array(self):
        t = self.u32()
        n = self.u64()
        return [self.value(t) for _ in range(n)]

def main(path):
    with open(path, "rb") as f:
        r = R(f)
        magic = f.read(4)
        assert magic == b"GGUF", magic
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        print(f"# GGUF v{version}: {n_tensors} tensors, {n_kv} metadata keys\n")
        print("## Metadata")
        for _ in range(n_kv):
            key = r.string()
            t = r.u32()
            val = r.value(t)
            if isinstance(val, list):
                head = val[:8]
                summary = f"array[{len(val)}] head={head}"
                if key.startswith("prism") or "hadamard" in key or "sign" in key:
                    c = Counter(val) if len(val) < 100000 else None
                    summary += f" counter={dict(c) if c and len(c) < 8 else 'n/a'}"
                if key.startswith("tokenizer"):
                    summary = f"array[{len(val)}] (tokenizer, omitted)"
                print(f"- `{key}`: {summary}")
            else:
                s = str(val)
                if len(s) > 200:
                    s = s[:200] + "..."
                print(f"- `{key}`: {s}")
        print("\n## Tensors")
        print("| name | shape (ne) | type | offset |")
        print("|---|---|---|---|")
        types = Counter()
        for _ in range(n_tensors):
            name = r.string()
            nd = r.u32()
            ne = [r.u64() for _ in range(nd)]
            t = r.u32()
            off = r.u64()
            tn = GGUF_TYPES.get(t, f"type{t}")
            types[tn] += 1
            print(f"| {name} | {ne} | {tn} | {off} |")
        print("\n## Type histogram")
        for k, v in types.most_common():
            print(f"- {k}: {v}")

if __name__ == "__main__":
    main(sys.argv[1])
