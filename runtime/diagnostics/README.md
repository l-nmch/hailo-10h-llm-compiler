# Diagnostics

Low-level tools for debugging a compiled LLM HEF on hardware, ordered from
"audit without a device" to "drive raw network groups". All of them talk to
the low-level `InferModel` API (the same layer genai's C++ server uses),
bypassing the genai Python wrapper.

| Tool | What it tells you |
|---|---|
| [hef_audit.py](hef_audit.py) | Is the HEF structurally correct? Groups, I/O shapes/formats, embedded resources (tokenizer/config/RoPE table) — no device needed |
| [hotpatch_hailo_config.py](hotpatch_hailo_config.py) | Fix an embedded `hailo-config.json` in place, no recompile |
| [manual_prefill_tbt_test.py](manual_prefill_tbt_test.py) | Where does on-chip behavior diverge from float32 HF — at prefill already, or only in token-by-token mode? Needs a reference NPZ from the compile machine |
| [generate_base_scope.py](generate_base_scope.py) | Does the *model* generate coherent text when the KV-cache is not involved? The control test that isolates cache issues |
| [kv_greedy.py](kv_greedy.py) | Does greedy generation through the real KV-cache path match float32 HF? Drives `__prefill` + `__tbt` like the genai server; `--mask-mode self` gives a cache-free tbt control |
| [runtime_inputs.py](runtime_inputs.py) | Shared helpers reproducing the host-side input conventions (mask layout, RoPE tables, uint16 embedding codes) |

## Recommended debugging order

1. `hef_audit.py` — confirm structure and that the embedded config carries
   the right keys (`prefill_input_tokens_count`, not `_size`).
2. `generate_base_scope.py` — if this is incoherent, the problem is in the
   compile itself, not the cache mechanism. It needs a HEF compiled with
   `pipeline/s6_compile_hef.py --include-base-scope` (the default HEF has
   only the two genai network groups — no base scope to drive). That variant
   is diagnostics-only: never serve it through hailo-ollama / genai.LLM().
3. `kv_greedy.py` — greedy generation through the real KV-cache path
   (`__prefill` + `__tbt`, driven exactly like the genai server); compare
   its ids with a float32 Hugging Face greedy run on the same prompt.
   `--mask-mode self` masks the whole cache: if that step matches HF's
   single-token forward while the normal mode does not, the problem is in
   what `__tbt` reads from the cache (the historic case, fixed in
   [../../docs/findings/tbt-cache-read.md](../../docs/findings/tbt-cache-read.md)).
4. `manual_prefill_tbt_test.py` — single prefill + tbt step compared
   against a reference NPZ at the hidden-state or logits level, for finer
   localization.

## Where `wte.npy` comes from

`generate_base_scope.py` (and any raw probe) needs the fp32 embedding table
saved by compile step 1 (`workdir/wte.npy`). The HEF already embeds the
embeddings — but that copy is there for the *runtime* (hailo-ollama /
genai.LLM do the token → embedding lookup host-side from
`embeddings.bin`). A raw probe drives `input_layer1` (`inputs_embeds`)
directly, so it must redo that lookup itself: token id → `wte[id]` → uint16
codes with `input_layer1`'s scale/zero-point. The same table also builds the
calibration set at compile time.

## Building the reference NPZ

`manual_prefill_tbt_test.py` compares against float32 activations exported
from the original Hugging Face model (run this where PyTorch is available):

```python
import numpy as np, torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("<model-id>", torch_dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained("<model-id>")

prompt_ids = tok("Once upon a time there was a little girl", return_tensors="pt")["input_ids"]
follow_up_id = <next token id you want to feed in tbt>

with torch.no_grad():
    out_prompt = model(prompt_ids, output_hidden_states=True)
    seq2 = torch.cat([prompt_ids, torch.tensor([[follow_up_id]])], dim=1)
    out_tbt = model(seq2, output_hidden_states=True)

np.savez("refs.npz",
    token_ids=prompt_ids.numpy().astype(np.int64)[0],
    next_token_id=int(out_prompt.logits[0, -1].argmax()),
    hf_hidden_prefill=out_prompt.hidden_states[-1][0, -1].numpy(),
    hf_hidden_tbt=out_tbt.hidden_states[-1][0, -1].numpy(),
    embed_rows_prompt=model.get_input_embeddings().weight[prompt_ids[0]].numpy(),
    embed_row_next=model.get_input_embeddings().weight[follow_up_id].numpy(),
)
```

The embedding quantization parameters (`--emb-qp-scale/--emb-qp-zp`) must
match YOUR quantized HAR's `input_layer1` parameters — read them once via
the SDK and pass them explicitly; the defaults are those of the reference
model of this repository.
