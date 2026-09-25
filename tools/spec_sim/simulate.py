"""Greedy speculative-decoding replay over recorded token sequences.

The "target model" emits the recorded completion. One round drafts D tokens from the context,
accepts the longest draft prefix equal to the recorded continuation (A tokens), and commits A + 1
tokens: the accepted drafts plus the target's correction or bonus token. This is exactly greedy
verification, so the committed text never changes; only the round count does.

Policies:
  mtp              fixed MTP-like proposer (k drafts, per-position conditional acceptance)
  ngram-simple     copy the m tokens after the previous occurrence of the last n tokens
  ngram-mod        walk the shared tagged hash pool
  select:<ngram>   MTP draft, replaced by the n-gram draft when that is longer than k tokens
                   and at least --draft-min
  chain:<ngram>    MTP draft followed by the n-gram continuation keyed on context + MTP draft
N-gram drafts (alone or as a chain extension) shorter than --draft-min are dropped.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .cost import CostModel
from .ngram import Context, NgramModPool, NgramSimpleIndex, Sink

NGRAM_KINDS = ("ngram-simple", "ngram-mod")
POLICIES = ("mtp", "ngram-simple", "ngram-mod", "select:ngram-simple", "select:ngram-mod",
            "chain:ngram-simple", "chain:ngram-mod")
ACCEPT_HISTOGRAM_EDGES = (0, 1, 2, 3, 4, 8, 16, 32, 64)
LONG_DRAFT = 15


@dataclass(frozen=True)
class Record:
    prompt: tuple[int, ...]
    completion: tuple[int, ...]
    label: str = ""


@dataclass(frozen=True)
class SimConfig:
    policy: str
    cap: int
    n: int = 12
    m: int = 64
    pool_bytes: int = 16 * 1024 * 1024
    tag_bits: int = 14
    draft_min: int = 1
    mtp_k: int = 3
    mtp_accept: tuple[float, ...] = (0.86,)
    pool_scope: str = "global"
    seed: int = 1234

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"unknown policy {self.policy!r}; choose from {', '.join(POLICIES)}")
        if self.cap < 1 or self.n < 1 or self.m < 1 or self.mtp_k < 1 or self.draft_min < 1:
            raise ValueError("cap, n, m, mtp_k and draft_min must be positive")
        if self.pool_scope not in ("global", "sequence"):
            raise ValueError("pool scope must be global or sequence")
        if not self.mtp_accept or any(not 0.0 <= p <= 1.0 for p in self.mtp_accept):
            raise ValueError("MTP acceptance probabilities must be in [0,1]")

    @property
    def uses_mtp(self) -> bool:
        return self.policy == "mtp" or self.policy.startswith(("select:", "chain:"))

    @property
    def ngram_kind(self) -> str | None:
        name = self.policy.split(":")[-1]
        return name if name in NGRAM_KINDS else None


@dataclass
class SimResult:
    config: SimConfig
    cost: CostModel
    records: int = 0
    rounds: int = 0
    tokens: int = 0
    drafted: int = 0
    accepted: int = 0
    draft_rounds: int = 0
    ngram_rounds: int = 0
    rounds_draft_over_15: int = 0
    rounds_accept_over_15: int = 0
    tokens_in_accept_over_15: int = 0
    total_ms: float = 0.0
    accept_counts: dict[int, int] = field(default_factory=dict)
    width_counts: dict[int, int] = field(default_factory=dict)

    def add_round(self, draft: int, accepted: int, committed: int, from_ngram: bool) -> None:
        width = self.cost.width_for(draft, self.config.cap)
        self.rounds += 1
        self.tokens += committed
        self.drafted += draft
        self.accepted += accepted
        self.draft_rounds += draft > 0
        self.ngram_rounds += from_ngram
        self.rounds_draft_over_15 += draft > LONG_DRAFT
        if accepted > LONG_DRAFT:
            self.rounds_accept_over_15 += 1
            self.tokens_in_accept_over_15 += committed
        self.total_ms += self.cost.verify_ms(width, draft, accepted)
        self.accept_counts[accepted] = self.accept_counts.get(accepted, 0) + 1
        self.width_counts[width] = self.width_counts.get(width, 0) + 1

    @property
    def tok_s(self) -> float:
        return 1000.0 * self.tokens / self.total_ms if self.total_ms else 0.0

    @property
    def tokens_per_round(self) -> float:
        return self.tokens / self.rounds if self.rounds else 0.0

    @property
    def accepted_per_round(self) -> float:
        return self.accepted / self.rounds if self.rounds else 0.0

    def accept_histogram(self) -> dict[str, int]:
        edges = ACCEPT_HISTOGRAM_EDGES
        buckets: dict[str, int] = {}
        for low, high in zip(edges, edges[1:] + (None,)):
            label = (str(low) if high == low + 1 else f"{low}+" if high is None
                     else f"{low}-{high - 1}")
            buckets[label] = sum(count for value, count in self.accept_counts.items()
                                 if value >= low and (high is None or value < high))
        return buckets

    def summary(self) -> dict:
        rounds = max(self.rounds, 1)
        return {
            "policy": self.config.policy,
            "cap": self.config.cap,
            "n": self.config.n,
            "records": self.records,
            "tokens": self.tokens,
            "rounds": self.rounds,
            "tokens_per_round": self.tokens_per_round,
            "accepted_per_round": self.accepted_per_round,
            "acceptance": self.accepted / self.drafted if self.drafted else 0.0,
            "draft_round_fraction": self.draft_rounds / rounds,
            "ngram_round_fraction": self.ngram_rounds / rounds,
            "draft_over_15_fraction": self.rounds_draft_over_15 / rounds,
            "accept_over_15_fraction": self.rounds_accept_over_15 / rounds,
            "token_share_accept_over_15": (self.tokens_in_accept_over_15 / self.tokens
                                           if self.tokens else 0.0),
            "ms": self.total_ms,
            "tok_s": self.tok_s,
            "speedup_vs_decode": self.tok_s / self.cost.decode_tok_s(),
            "accept_histogram": self.accept_histogram(),
            "width_histogram": {str(k): v for k, v in sorted(self.width_counts.items())},
        }


class MtpModel:
    """MTP-like fixed acceptance: draft j is the recorded token with conditional probability
    p_j given drafts < j were correct; the first miss and everything after it is wrong."""

    WRONG = -2

    def __init__(self, k: int, accept: Sequence[float], seed: int) -> None:
        self.k = k
        self.accept = tuple(accept[j] if j < len(accept) else accept[-1] for j in range(k))
        self.rng = random.Random(seed)

    def draft(self, truth: Sequence[int], start: int, limit: int) -> list[int]:
        out: list[int] = []
        correct = True
        for j in range(min(self.k, limit)):
            correct = correct and self.rng.random() < self.accept[j]
            out.append(truth[start + j] if correct else self.WRONG)
        return out


def common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    """Longest common prefix length, by slice comparison (C speed) with binary search."""
    low, high = 0, min(len(a), len(b))
    if list(a[:high]) == list(b[:high]):
        return high
    while low < high:
        mid = (low + high + 1) // 2
        if list(a[:mid]) == list(b[:mid]):
            low = mid
        else:
            high = mid - 1
    return low


class Simulator:
    def __init__(self, config: SimConfig, cost: CostModel) -> None:
        self.config = config
        self.cost = cost
        self.result = SimResult(config, cost)
        kind = config.ngram_kind
        self.pool = (NgramModPool.from_bytes(config.n, config.pool_bytes, config.tag_bits)
                     if kind == "ngram-mod" else None)
        self.mtp = MtpModel(config.mtp_k, config.mtp_accept, config.seed) if config.uses_mtp \
            else None
        self.context: Context | None = None
        self.simple: NgramSimpleIndex | None = None

    def _begin(self, prompt: Sequence[int]) -> Context:
        """Continue the previous lane history when the prompt extends it (prefix reuse),
        otherwise start a new history. The shared pool only receives unseen suffixes."""
        config = self.config
        previous = self.context
        if config.pool_scope == "sequence" and self.pool is not None:
            self.pool = NgramModPool.from_bytes(config.n, config.pool_bytes, config.tag_bits)
            previous = None
        shared = common_prefix(previous.tokens, prompt) if previous is not None else 0
        if previous is not None and shared == len(previous.tokens):
            context = previous
        else:
            context = Context(config.n)
            self.simple = None
            local = self._local_sinks(context)
            context.extend(prompt[:shared], local)
        context.extend(prompt[len(context.tokens):], self._sinks(context))
        self.context = context
        return context

    def _local_sinks(self, context: Context) -> list[Sink]:
        if self.config.ngram_kind == "ngram-simple":
            if self.simple is None:
                self.simple = NgramSimpleIndex(self.config.n, self.config.m)
            return [self.simple]
        return []

    def _sinks(self, context: Context) -> list[Sink]:
        sinks = self._local_sinks(context)
        if self.pool is not None:
            sinks.append(self.pool)
        return sinks

    def _ngram(self, context: Context, limit: int, prefix: Sequence[int] = ()) -> list[int]:
        proposer = self.pool if self.pool is not None else self.simple
        if proposer is None or limit <= 0:
            return []
        draft = proposer.propose(context, limit, prefix)
        return draft if len(draft) >= self.config.draft_min else []

    def _draft(self, context: Context, truth: Sequence[int], start: int,
               limit: int) -> tuple[list[int], bool]:
        policy = self.config.policy
        if limit <= 0:
            return [], False
        if policy in NGRAM_KINDS:
            draft = self._ngram(context, limit)
            return draft, bool(draft)
        assert self.mtp is not None
        mtp = self.mtp.draft(truth, start, limit)
        if policy == "mtp":
            return mtp, False
        if policy.startswith("select:"):
            ngram = self._ngram(context, limit)
            if len(ngram) > len(mtp):
                return ngram, True
            return mtp, False
        extension = self._ngram(context, limit - len(mtp), mtp)
        return mtp + extension, bool(extension)

    def run(self, record: Record) -> None:
        context = self._begin(record.prompt)
        sinks = self._sinks(context)
        truth = record.completion
        cap = self.config.cap
        position = 0
        self.result.records += 1
        while position < len(truth):
            remaining = len(truth) - position
            draft, from_ngram = self._draft(context, truth, position, min(cap, remaining - 1))
            accepted = 0
            while accepted < len(draft) and draft[accepted] == truth[position + accepted]:
                accepted += 1
            committed = accepted + 1
            context.extend(truth[position:position + committed], sinks)
            self.result.add_round(len(draft), accepted, committed, from_ngram)
            position += committed


def simulate(records: Iterable[Record], config: SimConfig, cost: CostModel) -> SimResult:
    simulator = Simulator(config, cost)
    for record in records:
        simulator.run(record)
    return simulator.result
