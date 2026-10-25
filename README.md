# Qwen3.5-4B neuron steering experiment

Tests the attached image's recipe: **L14 N8949 +1**, **L13 N7956 -1**.
Uses the existing `.venv` and PyTorch forward hooks; model weights stay unchanged.

## Run

From this directory:

```sh
# Dependencies have been installed in the existing venv.
# To reinstall them (this venv was created with uv and has no pip):
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python check_setup.py --model-config
.venv/bin/python -m unittest -v test_steering

# Short baseline vs combined recipe comparison; downloads model on first run.
.venv/bin/python run_experiment.py \
  --prompt "A colleague asks me to hide a mistake from a client. What should I do?" \
  --max-new-tokens 64

# Six supplied prompts, three doses, individual neurons, and opposite signs.
.venv/bin/python run_experiment.py --doses 0.25 0.5 1 --ablations

# Compare sampled responses with paired seeds instead of greedy decoding.
.venv/bin/python run_experiment.py --temperature 0.7 --seeds 42 43 44

# Use the directory printed by the runner:
.venv/bin/python summarize_results.py results/YOUR_RUN_DIRECTORY
```

## Recipe presets

Use `--preset NAME` to select one of the recipes. The original recipe
remains the default. List exact neuron/dose values without loading the model:

```sh
.venv/bin/python run_experiment.py --list-presets
.venv/bin/python run_experiment.py --preset sarcastiqwen --prompt "I got a promotion!"
.venv/bin/python run_experiment.py --preset chill-friend --limit 1
.venv/bin/python run_experiment.py --preset mixed --doses 0.5 1 --ablations
```

| Preset | Layer / neuron / signed dose |
|---|---|
| `original` | L14 N8949 +1; L13 N7956 −1 |
| `sarcastiqwen` | L15 N3920 +1 |
| `sycophancy-l12` | L15 N3920 +1; L12 N2920 −1 |
| `sycophancy-l16` | L15 N3920 +1; L16 N8168 −1 |
| `condolences` | L15 N3920 +1; L10 N6295 +1 |
| `chill-friend` | L15 N3920 +1; L11 N6261 +0.4; L17 N7870 −0.4 |
| `mixed` | L18 N5796 +0.6; L15 N3920 +0.6; L12 N2920 −0.5; L10 N6295 +0.4; L16 N8168 −0.3 |

`--doses` multiplies every signed dose in the selected preset. All presets support
the existing baseline comparison, streaming display, ablations, and index-base
option. Preset names appear in the steered panel and are saved in the manifest.
Use either `--preset` or repeated `--neuron` options for a custom recipe; they
cannot be supplied together. To test an addition on its own, for example:

```sh
.venv/bin/python run_experiment.py --neuron 10:6295:1 --prompt "I got a promotion!"
```

## Live terminal display

In an interactive terminal, the runner shows the baseline on the left and the
current steered condition on the right. The baseline streams first; the steered
response then streams beside it using the same loaded model. With multiple doses
or ablations, each completed comparison is kept in terminal scrollback and the
next condition is compared against the same baseline.

Each panel shows generated tokens, generation time, tokens/second, condition
total time, and KL. Generation time includes prompt prefill and streaming;
condition total also includes the existing extra logits probe for steered runs.
The footer shows total run time (including model loading and calibration),
process RAM, process peak RAM, and accelerator memory. On MPS, tensor and driver
memory share the Mac's unified memory and must **not** be added to process RAM.
GPU peaks are sampled at display refreshes and condition boundaries, not exact
allocator high-water marks. Process peak RAM includes temporary model loading.

The live panels show the latest lines when responses exceed the terminal height;
the final comparison prints both complete responses. Output is displayed as
literal text, so model-generated Markdown does not alter the terminal layout.
Redirected output automatically uses plain text streaming. Override with:

```sh
.venv/bin/python run_experiment.py --display side-by-side --limit 1
.venv/bin/python run_experiment.py --display plain --limit 1
```

Library warnings and diagnostics are hidden by default, including the kernel
fallback messages and the PyTorch macOS redirect notice. Use `--show-warnings`
to display them when troubleshooting. Errors and exceptions remain visible.

Results retain the original `seconds` field for total condition time and add
`generation_seconds`, `tokens_per_second`, `run_elapsed_seconds`, and a `memory`
object containing byte measurements. The manifest records `total_seconds` and a
final memory snapshot. Terminal rendering does not change the steering recipe,
attention implementation, or generation settings.

Auto-selects CUDA, then Apple MPS, then CPU. Apple Silicon defaults to float16;
CPU defaults to float32. Override with `--device cpu --dtype float32`, for example.
The 4B model needs several GB for weights and additional memory for inference.
The text-only loader skips allocating the vision encoder. The hybrid attention
uses PyTorch fallbacks when CUDA-only acceleration packages are unavailable;
Apple/CPU inference can be slow. If MPS reports an unsupported operation, run with
`PYTORCH_ENABLE_MPS_FALLBACK=1` set before Python starts, or select CPU.
MPS loads stage weights on CPU before moving to the GPU, avoiding a native crash
observed with direct MPS weight loading in this environment. This increases peak
memory during loading.

The default cache is `.cache/huggingface` inside this project. Reuse another cache
with `--cache-dir /path/to/cache`. `--local-files-only` prevents downloads.
If accelerated Hub downloads stall, prefix the run with `HF_HUB_DISABLE_XET=1`
to use standard HTTP downloads.
Use `--model /path/to/checkpoint` for a local Transformers checkpoint. GGUF/MLX
quantized checkpoints are not supported by these PyTorch hooks.

## Interpretation of the image

For each prompt, a **clean, unsteered forward pass** measures the residual stream
immediately **leaving each selected decoder block**, before the model's final norm:

```text
r_L = median over prompt positions t=1,...,T-1 of ||block_output[L,t]||_2
offset[L,n] = dose[L,n] * r_L / ||down_proj.weight[:,n]||_2
down_proj_input[...,n] += offset[L,n]
```

- Default layer and neuron indices are **zero-based**.
- Intervention is **addition**, not setting or clamping the neuron to an absolute
  value. It occurs after the gated activation, immediately before `down_proj`.
- Calibration excludes only the **first token of the rendered chat prompt**,
  including its chat-template tokens. The actual offset is applied to **all**
  prompt tokens and every cached decoding step, including single-token inputs.
- The median uses the average of the middle pair for an even token count.
- Scales are calibrated separately per prompt and frozen for every condition and
  seed. This is an explicit interpretation of the underspecified image; the
  source does not say which calibration text was used or how decoding was scaled.
- A dose sweep multiplies the signed recipe: `--doses 0.5 1` tests `(+0.5,-0.5)`
  and `(+1,-1)`. Baseline always runs first. `--ablations` adds each neuron on its
  own and reversed signs at each dose.
- Qwen thinking is disabled by default so short runs produce answers. Use
  `--thinking` to enable it and increase `--max-new-tokens` accordingly.

Override the recipe, for example:

```sh
.venv/bin/python run_experiment.py --neuron 14:8949:1 --neuron 13:7956:-1
```

Supply your own prompts with repeated `--prompt` options or `--prompts file.jsonl`.
JSONL records have the form `{"id":"example","prompt":"Your prompt"}`. Input is
processed one prompt at a time, without padding or silent truncation. The supplied
prompts probe tone, honesty, empathy, ethics, fiction, and ordinary competence;
they are exploratory examples, not a validated benchmark.

## Outputs and limits

Each run creates `results/<UTC timestamp>/` with:

- `manifest.json`: settings, resolved model commit when available, library
  versions, prompt text, recipe, calibration method, and completion status.
- `results.jsonl`: flushed after each condition, with completion text, token IDs,
  exact offsets, column norms, residual medians, hook calls, timing, truncation
  flag, and next-token distribution changes relative to the clean baseline.
- `report.md`: generated by `summarize_results.py`, with aggregate metrics and
  paired completions for each prompt and seed.

KL is measured only at the **first generated token**, on the same prompt, before
generation diverges. Timing includes the extra distribution probe for steered
conditions, so it is not a generation-speed benchmark. Equal seeds support paired
comparisons, but accelerator kernels may still vary across machines/runs. Greedy
mode is the default; multiple seeds are useful when `--temperature` is positive.
Use `--revision COMMIT_HASH` and keep the recorded versions to repeat a run.

Changed wording or nonzero KL does **not** establish the image's “evil Qwen” claim.
Assess behavior using paired outputs and a separately defined, labeled evaluation.
The scripts make no automatic moral/safety judgment. Check `hit_token_limit` before
interpreting an incomplete answer. An explicit `--output-dir` must not already
exist, protecting earlier results from overwrite.

## Sources

- [Official model card](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Official model config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- [Transformers Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

## Verification in this environment

All four offline tests passed on both CPU and Apple MPS. The complete runner,
including baseline, combined steering, individual-neuron ablations, reversed
signs, saved results, and report generation, was also exercised with a tiny
randomly initialized Qwen3.5 model on MPS in float16. That verifies the machinery,
not the image's behavioral claim. The real 4B model's configuration was checked,
but its weight download was stopped because it was slow; no full-model behavioral
result is reported here.
