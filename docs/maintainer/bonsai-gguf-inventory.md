# GGUF v3: 851 tensors, 49 metadata keys

## Metadata
- `general.architecture`: qwen35
- `general.type`: model
- `general.sampling.top_k`: 20
- `general.sampling.top_p`: 0.949999988079071
- `general.sampling.temp`: 1.0
- `general.name`: Hf
- `general.version`: v5
- `general.basename`: folded
- `general.size_label`: 27B
- `qwen35.block_count`: 64
- `qwen35.context_length`: 262144
- `qwen35.embedding_length`: 5120
- `qwen35.feed_forward_length`: 17408
- `qwen35.attention.head_count`: 24
- `qwen35.attention.head_count_kv`: 4
- `qwen35.rope.dimension_sections`: array[4] head=[11, 11, 10, 0]
- `qwen35.rope.freq_base`: 10000000.0
- `qwen35.attention.layer_norm_rms_epsilon`: 9.999999974752427e-07
- `qwen35.attention.key_length`: 256
- `qwen35.attention.value_length`: 256
- `qwen35.ssm.conv_kernel`: 4
- `qwen35.ssm.state_size`: 128
- `qwen35.ssm.group_count`: 16
- `qwen35.ssm.time_step_rank`: 48
- `qwen35.ssm.inner_size`: 6144
- `qwen35.full_attention_interval`: 4
- `qwen35.rope.dimension_count`: 64
- `prism.hadamard.version`: 1
- `prism.hadamard.block_size`: 1024
- `prism.hadamard.transform`: normalized-sylvester-walsh-hadamard
- `prism.hadamard.axis`: input-last-dimension
- `prism.hadamard.sign_mode`: explicit
- `prism.hadamard.weight_names`: array[401] head=['output.weight', 'blk.0.attn_qkv.weight', 'blk.0.attn_gate.weight', 'blk.0.ssm_out.weight', 'blk.0.ffn_down.weight', 'blk.0.ffn_gate.weight', 'blk.0.ffn_up.weight', 'blk.1.attn_qkv.weight'] counter=n/a
- `prism.hadamard.sign_widths`: array[3] head=[5120, 6144, 17408] counter={5120: 1, 6144: 1, 17408: 1}
- `prism.hadamard.sign_values`: array[28672] head=[-1, -1, -1, 1, -1, 1, 1, 1] counter={-1: 14504, 1: 14168}
- `prism.hadamard.inverse_weight_names`: array[1] head=['token_embd.weight'] counter={'token_embd.weight': 1}
- `prism.hadamard.gdn_v_grouped`: True
- `tokenizer.ggml.model`: gpt2
- `tokenizer.ggml.pre`: qwen35
- `tokenizer.ggml.tokens`: array[248320] (tokenizer, omitted)
- `tokenizer.ggml.token_type`: array[248320] (tokenizer, omitted)
- `tokenizer.ggml.merges`: array[247587] (tokenizer, omitted)
- `tokenizer.ggml.eos_token_id`: 248046
- `tokenizer.ggml.padding_token_id`: 248044
- `tokenizer.ggml.bos_token_id`: 248044
- `tokenizer.ggml.add_bos_token`: False
- `tokenizer.chat_template`: {%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, do_vision_count, is_system_content=false) %}
    {%- if content is string %}
 ...
- `general.quantization_version`: 2
- `general.file_type`: 143

## Tensors (blk.0 = GDN layer, blk.3 = full-attention layer; the other 62 layers repeat these two patterns)
| name | shape (ne) | type | offset |
|---|---|---|---|
| output.weight | [5120, 248320] | type143 | 0 |
| output_norm.weight | [5120] | F32 | 278118400 |
| token_embd.weight | [5120, 248320] | type143 | 278138880 |
| blk.0.attn_gate.weight | [5120, 6144] | type143 | 556257280 |
| blk.0.attn_norm.weight | [5120] | F32 | 563138560 |
| blk.0.attn_qkv.weight | [5120, 10240] | type143 | 563159040 |
| blk.0.ffn_down.weight | [17408, 5120] | type143 | 574627840 |
| blk.0.ffn_gate.weight | [5120, 17408] | type143 | 594124800 |
| blk.0.ffn_up.weight | [5120, 17408] | type143 | 613621760 |
| blk.0.post_attention_norm.weight | [5120] | F32 | 633118720 |
| blk.0.ssm_a | [48] | F32 | 633139200 |
| blk.0.ssm_alpha.weight | [5120, 48] | BF16 | 633139392 |
| blk.0.ssm_beta.weight | [5120, 48] | BF16 | 633630912 |
| blk.0.ssm_conv1d.weight | [4, 10240] | F32 | 634122432 |
| blk.0.ssm_dt.bias | [48] | F32 | 634286272 |
| blk.0.ssm_norm.weight | [128] | F32 | 634286464 |
| blk.0.ssm_out.weight | [6144, 5120] | type143 | 634286976 |
| blk.3.attn_k.weight | [5120, 1024] | type143 | 810990208 |
| blk.3.attn_k_norm.weight | [256] | F32 | 812137088 |
| blk.3.attn_norm.weight | [5120] | F32 | 812138112 |
| blk.3.attn_output.weight | [6144, 5120] | type143 | 812158592 |
| blk.3.attn_q.weight | [5120, 12288] | type143 | 819039872 |
| blk.3.attn_q_norm.weight | [256] | F32 | 832802432 |
| blk.3.attn_v.weight | [5120, 1024] | type143 | 832803456 |
| blk.3.ffn_down.weight | [17408, 5120] | type143 | 833950336 |
| blk.3.ffn_gate.weight | [5120, 17408] | type143 | 853447296 |
| blk.3.ffn_up.weight | [5120, 17408] | type143 | 872944256 |
| blk.3.post_attention_norm.weight | [5120] | F32 | 892441216 |

## Type histogram
- type143: 402
- F32: 353
- BF16: 96
