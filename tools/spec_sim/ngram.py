"""Draft-free n-gram proposers: a tagged hash pool and a previous-occurrence copier.

The hash, slot, tag and entry encoding are the exact contract of the C++ `NgramDraftPool`
(`src/models/qwen3_5/program/speculative/ngram_pool.h`). Both sides pin the same test vector, so
simulated pool collisions are the collisions the runtime pool would see.

Window hash:  h(w[0..n)) = sum_i u32(w[i]) * LCG^(n-1-i)  (mod 2^64), i.e. h = h*LCG + u32(token)
              over the window, which rolls in O(1) per appended token.
Mixed hash:   m = fmix64(h)  (MurmurHash3 64-bit finalizer).
Slot:         m mod entries.
Tag:          the top `tag_bits` bits of m (0 disables the check, the llama.cpp ngram-mod rule).
Entry:        u32 (tag << 18) | (token + 1); 0 is empty. Token ids must be below 2^18 - 1.
"""

from __future__ import annotations

from array import array
from typing import Sequence

MASK64 = (1 << 64) - 1
LCG_MULTIPLIER = 6364136223846793005
TOKEN_BITS = 18
TOKEN_MASK = (1 << TOKEN_BITS) - 1
MAX_TOKEN_ID = TOKEN_MASK - 1
DEFAULT_TAG_BITS = 32 - TOKEN_BITS
ENTRY_BYTES = 4


def u32(token: int) -> int:
    return token & 0xFFFFFFFF


def fmix64(value: int) -> int:
    value ^= value >> 33
    value = (value * 0xFF51AFD7ED558CCD) & MASK64
    value ^= value >> 33
    value = (value * 0xC4CEB9FE1A85EC53) & MASK64
    value ^= value >> 33
    return value


def window_hash(tokens: Sequence[int]) -> int:
    value = 0
    for token in tokens:
        value = (value * LCG_MULTIPLIER + u32(token)) & MASK64
    return value


def slot_of(window: int, entries: int) -> int:
    return fmix64(window) % entries


def tag_of(window: int, tag_bits: int) -> int:
    return fmix64(window) >> (64 - tag_bits) if tag_bits else 0


class Context:
    """One lane's token history plus the rolling hash of its last `n` tokens.

    `append` reports each new (window, next token, position) transition to the sinks before
    rolling, so a sink records exactly the n-grams whose continuation is known.
    """

    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError("n-gram length must be positive")
        self.n = n
        self.tokens: list[int] = []
        self.hash = 0
        self._drop = pow(LCG_MULTIPLIER, n, 1 << 64)

    def append(self, token: int, sinks: Sequence["Sink"] = ()) -> None:
        length = len(self.tokens)
        if length >= self.n:
            for sink in sinks:
                sink.record(self.hash, token, length, self)
            oldest = u32(self.tokens[length - self.n])
            self.hash = (self.hash * LCG_MULTIPLIER + u32(token) - oldest * self._drop) & MASK64
        else:
            self.hash = (self.hash * LCG_MULTIPLIER + u32(token)) & MASK64
        self.tokens.append(token)

    def extend(self, tokens: Sequence[int], sinks: Sequence["Sink"] = ()) -> None:
        for token in tokens:
            self.append(token, sinks)

    def rolled(self, prefix: Sequence[int]) -> int | None:
        """Hash of the window ending after `prefix` is appended, or None when it is short."""
        if len(self.tokens) + len(prefix) < self.n:
            return None
        value = self.hash
        length = len(self.tokens)
        if length < self.n:
            return window_hash((self.tokens + list(prefix))[-self.n:])
        for index, token in enumerate(prefix):
            position = length + index - self.n
            oldest = self.tokens[position] if position < length else prefix[position - length]
            value = (value * LCG_MULTIPLIER + u32(token) - u32(oldest) * self._drop) & MASK64
        return value


class Sink:
    def record(self, window: int, token: int, position: int, context: Context) -> None:
        raise NotImplementedError


class NgramModPool(Sink):
    """Fixed-size tagged hash pool shared by every lane (llama.cpp `ngram-mod` style)."""

    def __init__(self, n: int, entries: int, tag_bits: int = DEFAULT_TAG_BITS) -> None:
        if n <= 0 or entries <= 0:
            raise ValueError("pool needs positive n and entries")
        if not 0 <= tag_bits <= DEFAULT_TAG_BITS:
            raise ValueError(f"tag bits must be in [0,{DEFAULT_TAG_BITS}]")
        self.n = n
        self.entries = entries
        self.tag_bits = tag_bits
        self.table = array("I", bytes(ENTRY_BYTES * entries))
        self.occupied = 0

    @classmethod
    def from_bytes(cls, n: int, pool_bytes: int, tag_bits: int = DEFAULT_TAG_BITS) -> NgramModPool:
        return cls(n, max(1, pool_bytes // ENTRY_BYTES), tag_bits)

    def record(self, window: int, token: int, position: int, context: Context) -> None:
        if not 0 <= token <= MAX_TOKEN_ID:
            raise ValueError(f"token id {token} is outside the pool domain")
        mixed = fmix64(window)
        slot = mixed % self.entries
        tag = mixed >> (64 - self.tag_bits) if self.tag_bits else 0
        if self.table[slot] == 0:
            self.occupied += 1
        self.table[slot] = (tag << TOKEN_BITS) | (token + 1)

    def lookup(self, window: int) -> int | None:
        mixed = fmix64(window)
        entry = self.table[mixed % self.entries]
        if entry == 0:
            return None
        if self.tag_bits and (entry >> TOKEN_BITS) != (mixed >> (64 - self.tag_bits)):
            return None
        return (entry & TOKEN_MASK) - 1

    def propose(self, context: Context, limit: int, prefix: Sequence[int] = ()) -> list[int]:
        """Walk the pool from the window ending after `prefix`; return up to `limit` tokens."""
        if limit <= 0 or context.n != self.n:
            return []
        window = context.rolled(prefix)
        if window is None:
            return []
        combined_prefix = list(prefix)
        tokens = context.tokens
        length = len(tokens)
        n = self.n
        drop = context._drop
        out: list[int] = []
        table = self.table
        entries = self.entries
        tag_shift = 64 - self.tag_bits
        tag_bits = self.tag_bits
        while len(out) < limit:
            mixed = fmix64(window)
            entry = table[mixed % entries]
            if entry == 0 or (tag_bits and (entry >> TOKEN_BITS) != (mixed >> tag_shift)):
                break
            token = (entry & TOKEN_MASK) - 1
            position = length + len(combined_prefix) + len(out) - n
            if position < length:
                oldest = tokens[position]
            elif position - length < len(combined_prefix):
                oldest = combined_prefix[position - length]
            else:
                oldest = out[position - length - len(combined_prefix)]
            window = (window * LCG_MULTIPLIER + token - u32(oldest) * drop) & MASK64
            out.append(token)
        return out


class NgramSimpleIndex(Sink):
    """Per-lane map from an n-gram to the position after its most recent occurrence
    (llama.cpp `ngram-simple`): the draft copies the m tokens that followed it."""

    def __init__(self, n: int, m: int) -> None:
        if n <= 0 or m <= 0:
            raise ValueError("ngram-simple needs positive n and m")
        self.n = n
        self.m = m
        self.last: dict[int, int] = {}

    def record(self, window: int, token: int, position: int, context: Context) -> None:
        self.last[window] = position

    def propose(self, context: Context, limit: int, prefix: Sequence[int] = ()) -> list[int]:
        limit = min(limit, self.m)
        if limit <= 0 or context.n != self.n:
            return []
        window = context.rolled(prefix)
        if window is None:
            return []
        start = self.last.get(window)
        if start is None:
            return []
        tokens = context.tokens
        history = len(tokens)
        n = self.n
        combined_end = history + len(prefix)

        def at(index: int, out: list[int]) -> int:
            if index < history:
                return tokens[index]
            if index - history < len(prefix):
                return prefix[index - history]
            return out[index - combined_end]

        # Reject 64-bit hash collisions: the stored occurrence must equal the current window.
        if any(at(start - n + j, []) != at(combined_end - n + j, []) for j in range(n)):
            return []
        out: list[int] = []
        while len(out) < limit:
            # A source past the combined end reads tokens drafted in this call (LZ77-style
            # overlap), so a short period repeats for the whole draft.
            out.append(at(start + len(out), out))
        return out
