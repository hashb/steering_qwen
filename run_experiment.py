#!/usr/bin/env python3
"""Compare baseline, combined neuron steering, and optional ablations."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import math
from pathlib import Path
import random
import sys
import time
import warnings

# Configure the CLI before importing model libraries, which also log warnings
# during import. Library users retain control of their own logging settings.
if __name__ == "__main__" and "--show-warnings" not in sys.argv[1:]:
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from presets import PRESETS, list_presets
from steering import Neuron, Steering, calibrate, decoder_layers, select_device, select_dtype
from terminal_output import ComparisonDisplay, ComparisonStreamer, MemoryUsage

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / ".cache" / "huggingface"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default="main", help="Use a commit hash for an exact repeat.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--prompt", action="append", help="Repeat for multiple prompts; replaces --prompts.")
    parser.add_argument("--prompts", type=Path, default=ROOT / "prompts.jsonl")
    parser.add_argument("--limit", type=int, help="Only run the first N prompts.")
    parser.add_argument("--doses", type=float, nargs="+", default=[1.0],
                        help="Multipliers of the selected recipe; baseline always runs.")
    recipe_group = parser.add_mutually_exclusive_group()
    recipe_group.add_argument("--preset", choices=PRESETS, default="evilqwen",
                              help="Named recipe (default: evilqwen).")
    recipe_group.add_argument("--neuron", action="append", metavar="LAYER:NEURON:DOSE",
                              help="Use a custom recipe instead of a preset; repeat for each neuron.")
    parser.add_argument("--list-presets", action="store_true",
                        help="Print all preset neuron/dose values and exit without loading a model.")
    parser.add_argument("--index-base", type=int, choices=[0, 1], default=0,
                        help="Interpret BOTH layer and neuron labels with this index base.")
    parser.add_argument("--ablations", action="store_true", help="Also test each neuron alone and reversed signs.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=2048, help="Reject longer inputs; never silently truncate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy decoding.")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen thinking; default disabled.")
    parser.add_argument("--display", choices=["auto", "side-by-side", "plain"], default="auto",
                        help="Live side-by-side panels in a terminal; plain streaming when redirected.")
    parser.add_argument("--show-warnings", action="store_true",
                        help="Show library warnings and diagnostics (hidden by default).")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.max_input_tokens < 2 or (args.limit is not None and args.limit < 1):
        parser.error("Token limits and prompt limit must be positive (input limit >= 2).")
    if not math.isfinite(args.temperature) or args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("Require finite temperature >= 0 and 0 < top-p <= 1.")
    if not all(math.isfinite(x) for x in args.doses):
        parser.error("Doses must be finite.")
    return args


def read_prompts(args):
    if args.prompt:
        records = [{"id": f"custom-{i + 1}", "prompt": p} for i, p in enumerate(args.prompt)]
    else:
        records = []
        for number, line in enumerate(args.prompts.read_text().splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("prompt"), str):
                raise ValueError(f"{args.prompts}:{number}: expected an object with a prompt string.")
            records.append({"id": str(record.get("id", number)), "prompt": record["prompt"]})
    records = records[:args.limit]
    if not records or any(not r["prompt"].strip() for r in records):
        raise ValueError("At least one nonempty prompt is required.")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Prompt IDs must be unique.")
    return records


def make_recipe(args):
    if args.neuron:
        result = []
        for spec in args.neuron:
            layer, neuron, dose = spec.split(":")
            result.append(Neuron(int(layer), int(neuron), float(dose)))
    else:
        result = PRESETS[args.preset].neurons
    return tuple(Neuron(n.layer - args.index_base, n.neuron - args.index_base, n.dose) for n in result)


def conditions(recipe, args):
    result = [("baseline", tuple(Neuron(n.layer, n.neuron, 0.0) for n in recipe))]
    label = "combined" if args.neuron or args.preset == "evilqwen" else args.preset
    for multiplier in dict.fromkeys(args.doses):
        if multiplier == 0:
            continue
        scaled = tuple(Neuron(n.layer, n.neuron, n.dose * multiplier) for n in recipe)
        result.append((f"{label}_{multiplier:g}", scaled))
        if args.ablations:
            for n in scaled:
                result.append((f"only_L{n.layer}_N{n.neuron}_{multiplier:g}", (n,)))
            result.append((f"reversed_{multiplier:g}", tuple(Neuron(n.layer, n.neuron, -n.dose) for n in scaled)))
    return result


def top_tokens(logits, tokenizer, k=8):
    values, indices = logits.softmax(-1).topk(min(k, logits.numel()))
    return [{"id": int(i), "text": tokenizer.decode([int(i)]), "probability": float(p)}
            for p, i in zip(values, indices)]


def logit_metrics(baseline, changed):
    if not bool(torch.isfinite(changed).all()):
        raise RuntimeError("Non-finite steered logits; try a smaller dose or float32.")
    log_p, log_q = baseline.log_softmax(-1), changed.log_softmax(-1)
    return {"next_token_kl_baseline_to_condition": max(0.0, float((log_p.exp() * (log_p - log_q)).sum())),
            "next_token_argmax_changed": bool(baseline.argmax() != changed.argmax()),
            "next_token_max_abs_logit_delta": float((baseline - changed).abs().max())}


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def main():
    run_started = time.perf_counter()
    args = parse_args()
    if args.list_presets:
        print(list_presets())
        return
    prompts, recipe = read_prompts(args), make_recipe(args)
    variants = conditions(recipe, args)
    device = select_device(args.device)
    dtype = select_dtype(device, args.dtype)
    memory = MemoryUsage(device)
    memory.snapshot()
    output_dir = args.output_dir or ROOT / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    load_args = dict(revision=args.revision, cache_dir=str(args.cache_dir),
                     local_files_only=args.local_files_only, trust_remote_code=False)
    print(f"Loading {args.model} on {device} ({dtype})\nResults: {output_dir}", flush=True)
    set_seed(args.seeds[0])
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_args)
    # Transformers extracts the text config and converts the VLM checkpoint's
    # language_model prefix. No vision encoder is allocated for this experiment.
    # Loading safetensors straight onto MPS can crash in native code on some
    # PyTorch/macOS combinations. Stage on CPU, then transfer in one step.
    load_device = "cpu" if device == "mps" else device
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="eager",
                                                device_map=load_device, **load_args)
    if device == "mps":
        model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    layers = decoder_layers(model)
    # Validate indices before doing any inference.
    from steering import validate_recipe
    validate_recipe(layers, recipe)
    manifest = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "device": device, "dtype": str(dtype), "model_commit": getattr(model.config, "_commit_hash", None),
                "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "accelerate")},
                "python": sys.version, "recipe_zero_based": [asdict(n) for n in recipe],
                "preset": None if args.neuron else args.preset,
                "calibration": "per-prompt clean block-output L2 median over tokens[1:], frozen during generation",
                "intervention": "add to down_proj input, all prefill and decode tokens",
                "conditions": [name for name, _ in variants], "prompts": prompts}
    manifest_path = output_dir / "manifest.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save_manifest()
    try:
        with (output_dir / "results.jsonl").open("w", buffering=1) as result_file, torch.inference_mode():
            for prompt in prompts:
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt["prompt"]}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=args.thinking)
                inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)
                prompt_length = inputs["input_ids"].shape[1]
                if prompt_length > args.max_input_tokens:
                    raise ValueError(f"{prompt['id']}: {prompt_length} input tokens exceeds limit.")
                medians, clean_logits = calibrate(model, inputs, recipe)
                baseline_tokens = {}
                for seed in dict.fromkeys(args.seeds):
                    first_condition = variants[1][0] if len(variants) > 1 else None
                    with ComparisonDisplay(prompt, seed, first_condition, memory, run_started,
                                           mode=args.display) as display:
                        for name, variant in variants:
                            intervention = Steering(model, variant, medians)
                            set_seed(seed)
                            random.seed(seed)
                            synchronize(device)
                            start = time.perf_counter()
                            display.begin(name)
                            with intervention.applied():
                                changed_logits = clean_logits if name == "baseline" else model(
                                    **inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu()
                                metrics = logit_metrics(clean_logits, changed_logits)
                                generation_args = dict(max_new_tokens=args.max_new_tokens, use_cache=True,
                                                       do_sample=args.temperature > 0,
                                                       pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                                                       streamer=ComparisonStreamer(tokenizer, display, name))
                                if args.temperature > 0:
                                    generation_args.update(temperature=args.temperature, top_p=args.top_p)
                                synchronize(device)
                                generation_start = display.start_generation(name)
                                generated = model.generate(**inputs, **generation_args)
                            synchronize(device)
                            generation_seconds = time.perf_counter() - generation_start
                            elapsed = time.perf_counter() - start
                            new_tokens = generated[0, prompt_length:].cpu().tolist()
                            if name == "baseline":
                                baseline_tokens[seed] = new_tokens
                            if any(n.dose for n in variant) and not intervention.calls:
                                raise RuntimeError("Steering hooks were never invoked.")
                            record = {"prompt_id": prompt["id"], "prompt": prompt["prompt"],
                                      "condition": name, "seed": seed, "input_tokens": prompt_length,
                                      "generated_tokens": len(new_tokens), "generated_token_ids": new_tokens,
                                      "completion": tokenizer.decode(new_tokens, skip_special_tokens=True),
                                      "completion_with_special_tokens": tokenizer.decode(new_tokens, skip_special_tokens=False),
                                      "same_tokens_as_baseline": new_tokens == baseline_tokens[seed],
                                      "hit_token_limit": len(new_tokens) == args.max_new_tokens,
                                      "seconds": elapsed, "generation_seconds": generation_seconds,
                                      "tokens_per_second": len(new_tokens) / generation_seconds,
                                      "run_elapsed_seconds": time.perf_counter() - run_started,
                                      "memory": memory.snapshot(), "interventions": intervention.entries,
                                      "hook_calls_by_layer": intervention.calls,
                                      "next_token_top": top_tokens(changed_logits, tokenizer), **metrics}
                            result_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            display.finish(record)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=repr(exc))
        raise
    finally:
        manifest["total_seconds"] = time.perf_counter() - run_started
        manifest["memory"] = memory.snapshot()
        save_manifest()
    print(f"\nTotal run time: {manifest['total_seconds']:.2f}s (includes model loading and calibration)")
    print(memory.text())
    print(f"\nCompare outputs: .venv/bin/python summarize_results.py {output_dir}")


if __name__ == "__main__":
    main()
