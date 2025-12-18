# Weight Tying Fix for Pi0.5 Model Loading

## The Problem

When loading a fine-tuned Pi0.5 model, you may see this warning:

```
Missing key(s) in state_dict: "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
```

This causes the model's language understanding to be broken because the embedding layer (which converts text tokens to vectors) is missing its weights.

## Root Cause: Weight Tying

Pi0.5 uses PaliGemma as its vision-language backbone. PaliGemma (like many transformer models) uses **weight tying** - a memory optimization where two layers share the same weights:

1. **`embed_tokens`** - Input embedding layer (converts token IDs → vectors)
2. **`lm_head`** - Output projection layer (converts vectors → token logits)

These layers perform inverse operations, so sharing weights makes sense mathematically and saves ~50% of embedding memory.

### How Weight Tying Works

```
Token IDs ──► embed_tokens.weight ──► Hidden States ──► lm_head.weight ──► Logits
                    │                                         │
                    └─────────── SAME TENSOR ─────────────────┘
```

## Why the Key is Missing

During fine-tuning with frameworks like OpenPI, the model saves weights to disk. Because `embed_tokens` and `lm_head` are tied (same tensor), some save implementations only save one copy to avoid duplication.

**What gets saved:**
```
model.paligemma_with_expert.paligemma.lm_head.weight  ✓ Saved
```

**What's expected but missing:**
```
model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight  ✗ Missing
```

When loading, the code expects both keys, finds only one, and leaves `embed_tokens` randomly initialized.

## The Impact

Without proper embeddings:
- The model cannot understand text instructions ("Pick up the lego")
- It produces essentially random behavior
- The robot moves but doesn't accomplish the task

## The Fix

We added code to `modeling_pi05openpi.py` that detects this situation and copies the `lm_head` weight to `embed_tokens` before loading:

```python
# Fix weight tying: PaliGemma ties embed_tokens with lm_head
# During fine-tuning, only lm_head may be saved, so we need to copy it to embed_tokens
embed_tokens_key = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
lm_head_key = "model.paligemma_with_expert.paligemma.lm_head.weight"

if embed_tokens_key not in remapped_state_dict and lm_head_key in remapped_state_dict:
    print(f"Fixing weight tying: copying {lm_head_key} -> {embed_tokens_key}")
    remapped_state_dict[embed_tokens_key] = remapped_state_dict[lm_head_key]
```

## Verification

After the fix, you should see:
```
Fixing weight tying: copying model.paligemma_with_expert.paligemma.lm_head.weight -> model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight
```

And the "Missing key(s)" warning should disappear.

## Technical Details

| Key | Shape | Description |
|-----|-------|-------------|
| `paligemma.lm_head.weight` | (vocab_size, hidden_dim) | Output projection |
| `paligemma.model.language_model.embed_tokens.weight` | (vocab_size, hidden_dim) | Input embedding |

Both have the same shape because they're designed to be tied.

## Files Modified

- `src/lerobot/policies/pi05_openpi/modeling_pi05openpi.py` - Added weight tying fix in `from_pretrained()` method

## Related Issues

This is a common issue when:
- Fine-tuning models that use weight tying
- Converting models between frameworks (JAX → PyTorch)
- Using checkpoints saved with different serialization settings

