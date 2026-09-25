"""Round-cost model: milliseconds per verification round as a function of verify width T.

T is the number of target columns in one round (draft tokens + 1). The default anchors are
single-lane Qwen3.8-27B measurements on the RTX 4090 (WINDOWS_PORT.md):

    T = 1   21.3 ms   plain greedy decode (tg128, 47 tok/s)
    T = 4   25.9 ms   MTP 3 round, MTP head included
    T = 13  33.5 ms   DFlash2 d12 round, 3.5 ms drafter included

Between anchors the cost is linear. Past the last anchor each extra column costs the prefill
marginal cost, 1/2000 s at ~2000 tok/s (pp2048 2035 tok/s), so C(33) = 43.5 ms and
C(65) = 59.5 ms. Proposer overheads are not separated: the anchors already contain an MTP head
or a drafter, so the curve is conservative for rounds whose draft comes from the host pool.

Rounds are costed at a graph width, not at the exact draft length: CUDA Graph profiles are
captured for a finite set of widths, and a round runs the smallest captured width that holds its
draft. `width_for` models that set.

Wide verification has two candidate implementations (docs/maintainer/ngram-speculation-plan.md).
`records` (default) widens the ReplaySSM record/fold path, so a round costs C(T) whatever it
accepts. `replay` runs rounds wider than `replay_above` through the prefill route into a scratch
state slot; a partially accepted round then re-runs its committed tokens from the old state and
costs C(T) + C(A + 1).
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Sequence

DEFAULT_POINTS: tuple[tuple[int, float], ...] = ((1, 21.3), (4, 25.9), (13, 33.5))
DEFAULT_MARGINAL_MS = 0.5
DEFAULT_WIDTH_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 33, 48, 65)


@dataclass(frozen=True)
class CostModel:
    points: tuple[tuple[int, float], ...] = DEFAULT_POINTS
    marginal_ms: float = DEFAULT_MARGINAL_MS
    width_buckets: tuple[int, ...] = field(default=DEFAULT_WIDTH_BUCKETS)
    # 0 selects the records model; otherwise widths above this use the prefill-replay model.
    replay_above: int = 0

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("cost model needs at least one anchor")
        widths = [t for t, _ in self.points]
        if widths != sorted(set(widths)) or widths[0] < 1:
            raise ValueError("cost anchors need strictly increasing positive widths")
        if self.marginal_ms < 0 or any(ms <= 0 for _, ms in self.points):
            raise ValueError("costs must be positive")
        buckets = list(self.width_buckets)
        if buckets != sorted(set(buckets)) or not buckets or buckets[0] != 1:
            raise ValueError("width buckets must be strictly increasing and start at 1")

    def round_ms(self, width: int) -> float:
        if width < 1:
            raise ValueError("verify width must be positive")
        points = self.points
        if width <= points[0][0]:
            return points[0][1]
        for (t0, c0), (t1, c1) in zip(points, points[1:]):
            if width <= t1:
                return c0 + (c1 - c0) * (width - t0) / (t1 - t0)
        last_t, last_c = points[-1]
        return last_c + (width - last_t) * self.marginal_ms

    def width_for(self, draft_tokens: int, cap: int) -> int:
        """Smallest captured width holding `draft_tokens`; cap + 1 is always captured."""
        need = draft_tokens + 1
        buckets = sorted({b for b in self.width_buckets if b <= cap + 1} | {cap + 1})
        return buckets[bisect_left(buckets, need)]

    def verify_ms(self, width: int, drafted: int, accepted: int) -> float:
        """Cost of one round at `width` that accepted `accepted` of `drafted` tokens."""
        cost = self.round_ms(width)
        if self.replay_above and width > self.replay_above and accepted < drafted:
            cost += self.round_ms(accepted + 1)
        return cost

    def decode_tok_s(self) -> float:
        return 1000.0 / self.round_ms(1)


def parse_points(text: str) -> tuple[tuple[int, float], ...]:
    points = []
    for item in text.split(","):
        width, _, ms = item.partition(":")
        points.append((int(width), float(ms)))
    return tuple(sorted(points))


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split(",") if item.strip())


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(item) for item in text.split(",") if item.strip())


def describe(model: CostModel, widths: Sequence[int] = (1, 4, 13, 16, 33, 65)) -> str:
    return ", ".join(f"C({w})={model.round_ms(w):.1f}ms" for w in widths)
