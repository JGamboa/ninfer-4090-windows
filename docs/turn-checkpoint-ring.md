# Turn checkpoint ring

> **Retired.** `--turn-checkpoints` is accepted and ignored since the 2026-09-01 upstream
> merge; the ring below no longer exists. Its job - restoring below a mid-history edit -
> is done by upstream's private long anchors, which this fork places automatically at the
> last N message boundaries of every prompt with `--auto-long-anchors N` (default: the
> `--max-long-anchors-per-continuation` cap). Without that flag no anchor is ever
> captured, because no HTTP protocol can place the explicit marker upstream expects. Size
> `--host-state-slots` for `continuations x (2 + anchors)` state images of about 147 MiB.
> See [serving.md](serving.md). The rest of this page is kept as history.

`--turn-checkpoints N` keeps up to N past turn checkpoints per slot in host memory.
A prompt that diverges from the resident session in the middle of its history then
restores at the deepest checkpoint below the divergence point. The server
re-prefills only the suffix after that checkpoint. Without the ring, the same
prompt re-prefills from token zero.

The flag defaults to `0` (off). The recommended production value on Qwen3.8-27B is
`32`. The policy mirrors llama.cpp's `--ctx-checkpoints`, adapted to this engine's
hybrid attention.

## Why the engine needs checkpoints at all

Qwen3.8-27B is a hybrid model: 16 of its 64 layers use full attention, the other
48 use GDN linear attention. The two halves age differently when a prompt edits
history:

- Full-attention KV is positional. The engine can truncate it to any frontier and
  the entries below the edit stay valid.
- Each GDN layer keeps one running recurrent state per slot. That state integrates
  every token in order and cannot rewind. The engine can resume only at a point
  where the state was copied aside.

Without the ring, the engine holds exactly one such copy: the turn checkpoint at
the start of the current turn. A prompt that edits anything before that point
forces a full re-prefill, whatever its depth. The ring retains the older copies
that the single checkpoint used to overwrite.

## What one checkpoint holds

| Component | Content | Size on Qwen3.8-27B |
|---|---|---:|
| Recurrent state | 48 layers x 128x128x48 fp32 | 144 MiB |
| Convolution state | 48 layers x 10240x3 bf16 | 2.9 MiB |
| Boundary hidden | 5120 bf16 | 10 KiB |

One entry therefore costs about 147 MiB of pageable host memory. The attention KV
is not copied. An entry stays valid only while the token ledger below its frontier
is untouched, so the slot's paged KV still holds those positions on the GPU.

## How to run it

Add the flag to any serve line:

```bash
ninfer-serve models/qwen3_8_27b.ninfer ... --turn-checkpoints 32
```

Host memory cost is `N x 147 MiB x --max-concurrency`, plus one fixed pinned
staging entry of 147 MiB per slot. GPU memory is unchanged.

| `--turn-checkpoints` | Host memory per slot | History covered |
|---:|---:|---:|
| 8 | 1.2 GiB | ~33K tokens |
| 32 | 4.6 GiB | ~131K tokens |
| 64 | 9.2 GiB | full 262K context |

The history coverage follows from the compaction spacing described below. Values
above 64 waste memory: the spacing can never retain more than
`context / 4096 = 64` entries.

## Capture and compaction

The engine captures a checkpoint when prefill crosses the current turn boundary,
which is the end of the last user message. The copy to host runs asynchronously on
the engine stream at a point where the stream is idle, so capture does not delay
the first token. The next request for the slot folds the staged copy into the
ring, or discards it when that request rewrote history below the staged frontier.

Ring compaction runs at each fold:

1. An entry with the same frontier is replaced.
2. Entries within 4,096 tokens of a deeper neighbour are folded away
   (`kTurnCheckpointMinStep`). The newest entry always survives.
3. Above capacity, the oldest entry is evicted.

Consequence: retained entries sit at least 4,096 tokens apart, so N entries cover
about `N x 4096` tokens of rewindable history. Short consecutive turns share one
checkpoint; a rewind to an unretained turn restores at the nearest deeper entry
and re-prefills the gap.

## Restore

Reuse planning tries three sources in order:

1. Exact frontier extension (`AppendAtFrontier`).
2. The resident turn checkpoint (`RestoreTurnCheckpoint`).
3. The ring, deepest matching entry first.

A ring restore uploads the entry back into the device checkpoint slot and then
follows the ordinary `RestoreTurnCheckpoint` path: the KV is truncated to the
entry's frontier, the GDN state is rewound to the copy, and the suffix is
re-prefilled. Generation after a ring restore is identical to a cold prefill of
the same prompt; the end-to-end test verifies greedy-exact output on the real
artifact.

## Observability

`GET /slots` lists each retained slot's checkpoints, oldest first:

```json
"checkpoints": [
  {"frontier": 12480, "session_digest": "a1b2c3d4e5f60718"},
  {"frontier": 30976, "session_digest": "18f7e6d5c4b3a291"}
]
```

`frontier` is the ledger depth the entry rewinds to. `session_digest` is the
FNV-1a 64 hash of the token ledger up to that frontier, in the same encoding as
the slot's `session_digest`. Two slots that report the same checkpoint digest hold
an identical history up to that point.

## Persistence and eviction

Slot snapshots (`--slot-save-path`) carry the ring. A snapshot with ring entries
is written as format version 2; a snapshot with an empty ring stays version 1, so
binaries without ring support keep reading their existing files. A restore into a
server with a smaller ring keeps the newest entries that fit.

The ring lives and dies with slot residency. Anything that evicts the resident
session also discards its ring:

- admission of a new session when every slot is retained (the cheapest slot is
  evicted: an empty one first, then the shallowest),
- `POST /slots/{id}?action=erase` or a restore over the slot,
- a server restart.

The snapshot path is the tolerance mechanism for slot churn. When clients rotate
more sessions than `--max-concurrency` slots, save each session to disk before it
loses its slot; the ring returns with the restore. A session evicted without a
snapshot starts cold and rebuilds its ring one turn per request.

`--auto-save-evicted` closes the remaining gap: before an involuntary eviction
destroys a retained session, the server spills it (ring included) back to the
slot file it was last saved to or restored from. Sessions that never touched a
slot file are not covered, and an explicit `erase` never auto-saves. The device
snapshot runs on the eviction path and costs a few hundred milliseconds; the
file write runs on a background thread. Requires `--slot-save-path`.

## Limits

- The DFlash backend is not supported. Its cyclic local cache mirrors only the
  resident checkpoint and cannot rebuild older entries.
- Checkpoints exist at turn boundaries only. An edit inside the first turn, or
  before the oldest retained entry, still takes a full re-prefill.
- A ring restore is not free: the suffix between the checkpoint and the new
  frontier is re-prefilled, and the 147 MiB upload from pageable memory costs
  roughly 15 ms before the prefill starts.
- Tool or system-prompt edits diverge near token zero, below every checkpoint.
  The ring cannot help there; keep volatile content out of the prompt prefix
  instead.
