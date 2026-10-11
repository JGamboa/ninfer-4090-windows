# MTP self-distillation

Retrains a model's MTP draft layer against the model's own predictions, so the drafts match the
quantized model the engine actually verifies. The main model is not changed: only the 13 MTP
objects of the artifact are rewritten, so outputs stay those of the target model (verification is
lossless) and every draft round costs the same. Measured on Ternary Bonsai 2 27B (RTX 4090,
2026-10-11): +11.8 % greedy decode, +15.7 % sampled with thinking (Bonsai design notes, section
9.1, item 37).

## Pipeline

All commands run from the repository root with the converter's Python environment
(`.venv\Scripts\python.exe` on Windows); paths are examples.

1. **Prompts.** Code problems from `ise-uiuc/Magicoder-OSS-Instruct-75K`
   (`data-oss_instruct-decontaminated.jsonl`, MIT), English and Spanish user prompts from
   `OpenAssistant/oasst2` (`2023-11-05_oasst2_prompts.messages.jsonl.gz`, Apache-2.0), plus
   built-in code-edit, JSON and math templates:

   ```powershell
   python -m tools.mtp_distill.build_prompts E:\LLM\mtp-train\prompts_full.jsonl `
     --oasst <oasst2 prompts .jsonl.gz> --magicoder <magicoder .jsonl>
   ```

2. **Completions.** Start `ninfer-serve` with the target artifact (model-card sampling defaults,
   MTP for speed) and let the target answer every prompt; about 30 % use thinking. Each line of
   the output is the chat-rendered prompt plus the target's reasoning and answer:

   ```powershell
   python -m tools.mtp_distill.gen_data http://127.0.0.1:8090/v1 E:\LLM\mtp-train\full\texts.jsonl `
     --prompts E:\LLM\mtp-train\prompts_full.jsonl --threads 3
   ```

3. **Captures.** `ninfer-score-export` scores every text with `Engine::score_tokens` and writes,
   per position, the final-norm hidden state the MTP layer consumes and the target's argmax
   (`--kv-dtype` should match the serving KV type):

   ```powershell
   build\apps\ninfer-score-export.exe E:\LLM\bonsai2_27b_vl_mtp_q4q5.ninfer `
     --input E:\LLM\mtp-train\full\texts.jsonl --output E:\LLM\mtp-train\full\export `
     --max-context 8192 --kv-dtype rk4v4-e8
   ```

4. **Training.** `train` reproduces the MTP layer in PyTorch (`mtp_torch.py`, a line-by-line
   port of `src/models/qwen3_5/execution/text.cpp`) and trains it with cross-entropy against the
   target's argmax over the frozen proposal head. Three chained draft steps are teacher-forced
   the way the runtime drafts them (step `s` takes step `s-1`'s output at position `t+s-1` and
   attends to the verified MTP KV plus its own drafts); the loss weights are 1, 0.6 and 0.4.
   Every tenth text (by id hash) is held out:

   ```powershell
   python -m tools.mtp_distill.train train --artifact E:\LLM\bonsai2_27b_vl_mtp_q4q5.ninfer `
     --embedding-cache E:\LLM\mtp-train\bonsai_embedding.pt --data E:\LLM\mtp-train\full\export `
     --weights E:\LLM\qwen3_8_27b_a8.ninfer --out E:\LLM\mtp-train\full_chain3.pt `
     --steps 3 --lr 1e-5 --epochs 1 --max-columns 3072 --eval-every 1000
   ```

   `--weights` initializes the layer; the Q8 MTP of a Qwen3.8 artifact is the most precise copy
   of the original weights. `eval` instead of `train` reports the held-out accuracy of each
   draft step for any weights.

5. **Artifact.** `patch_mtp` copies the target artifact and rewrites only its MTP objects,
   requantized in their stored formats with the converter's `quantize_matrix` and
   `encode_row_split`. `--check` verifies that re-encoding the original objects reproduces them
   byte for byte:

   ```powershell
   python -m tools.mtp_distill.patch_mtp E:\LLM\bonsai2_27b_vl_mtp_q4q5.ninfer `
     E:\LLM\bonsai2_27b_vl_mtp_retrained.ninfer E:\LLM\mtp-train\full_chain3.pt
   ```

Then compare decode speed against the original artifact on the same prompts and binary.

## Cost (Bonsai, RTX 4090)

3,342 prompts gave 2.98 M completion tokens in 2.0 h (three lanes, ~410 tok/s); the export
took 11 min (33 GB of hidden states for 3.49 M tokens); training one epoch took 16 min with
~14 GB of GPU memory. The decoded embedding cache is 2.5 GB.
