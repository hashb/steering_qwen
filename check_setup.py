#!/usr/bin/env python3
"""Check the active Python, device, and optional model config (no weights)."""

import argparse
import importlib.metadata
from pathlib import Path
import shutil
import sys

import psutil
import torch
from transformers import AutoConfig, Qwen3_5ForCausalLM

from steering import DEFAULT_RECIPE, select_device, select_dtype


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", action="store_true", help="Download the small model config only.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent / ".cache/huggingface")
    args = parser.parse_args()
    print(f"Python: {sys.executable}\nVersion: {sys.version.split()[0]}")
    for package in ("torch", "transformers", "accelerate", "huggingface_hub"):
        print(f"{package}: {importlib.metadata.version(package)}")
    print(f"Qwen text class: {Qwen3_5ForCausalLM.__name__}")
    print(f"CUDA: {torch.cuda.is_available()} | MPS: {torch.backends.mps.is_available()}")
    device = select_device()
    print(f"Default device/dtype: {device} / {select_dtype(device)}")
    print(f"System RAM: {psutil.virtual_memory().total / 2**30:.1f} GiB")
    print(f"Free workspace disk: {shutil.disk_usage(Path(__file__).resolve().parent).free / 2**30:.1f} GiB")
    if args.model_config:
        config = AutoConfig.from_pretrained(args.model, cache_dir=str(args.cache_dir), trust_remote_code=False)
        text = config.get_text_config()
        print(f"Model: {args.model}\nLayers: {text.num_hidden_layers} | MLP width: {text.intermediate_size}")
        for n in DEFAULT_RECIPE:
            if not (n.layer < text.num_hidden_layers and n.neuron < text.intermediate_size):
                raise ValueError(f"Recipe does not fit this config: {n}")
        print("The image's zero-based layer/neuron indices fit the model config.")


if __name__ == "__main__":
    main()
