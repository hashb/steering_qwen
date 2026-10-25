"""Additive MLP-neuron interventions; no model weights are modified."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class Neuron:
    layer: int
    neuron: int
    dose: float


DEFAULT_RECIPE = (Neuron(14, 8949, 1.0), Neuron(13, 7956, -1.0))


def decoder_layers(model):
    """Find the text decoder in text-only and multimodal Qwen wrappers."""
    for path in ("model.layers", "model.language_model.layers", "language_model.layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
        if obj is not None and len(obj) and hasattr(obj[0], "mlp"):
            return obj
    raise ValueError("Cannot find Qwen text decoder layers on this model.")


def validate_recipe(layers, recipe):
    seen = set()
    for item in recipe:
        if not math.isfinite(item.dose):
            raise ValueError("Doses must be finite.")
        if not 0 <= item.layer < len(layers):
            raise ValueError(f"Layer {item.layer} outside [0, {len(layers) - 1}].")
        proj = layers[item.layer].mlp.down_proj
        if not isinstance(proj, torch.nn.Linear):
            raise ValueError("Use unquantized torch.nn.Linear down_proj modules.")
        if not 0 <= item.neuron < proj.in_features:
            raise ValueError(f"Neuron {item.neuron} outside [0, {proj.in_features - 1}].")
        key = (item.layer, item.neuron)
        if key in seen:
            raise ValueError(f"Duplicate neuron: {key}.")
        seen.add(key)


@torch.inference_mode()
def calibrate(model, inputs, recipe):
    """Clean prompt pass: median L2 norm of each block's output, tokens 1:.

    Batch size is deliberately one and padding is disallowed. The median uses
    the average of the two middle values for an even number of tokens.
    """
    ids = inputs["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 2:
        raise ValueError("Calibration requires one prompt with at least two tokens.")
    if "attention_mask" in inputs and not bool(inputs["attention_mask"].all()):
        raise ValueError("Pass one unpadded prompt at a time.")
    layers = decoder_layers(model)
    validate_recipe(layers, recipe)
    medians, handles = {}, []

    def capture(index):
        def hook(module, args, output):
            residual = output[0] if isinstance(output, tuple) else output
            norms = residual[0, 1:].detach().float().norm(dim=-1)
            # CPU avoids MPS quantile limitations; only one scalar per token.
            value = float(torch.quantile(norms.cpu(), 0.5))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid residual norm at layer {index}: {value}.")
            medians[index] = value
        return hook

    try:
        for index in sorted({item.layer for item in recipe}):
            handles.append(layers[index].register_forward_hook(capture(index)))
        output = model(**inputs, use_cache=False, logits_to_keep=1)
        logits = output.logits[0, -1].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()

    if len(medians) != len({item.layer for item in recipe}):
        raise RuntimeError("Some calibration hooks did not run.")
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Non-finite baseline logits.")
    return medians, logits


class Steering:
    def __init__(self, model, recipe, medians):
        self.layers = decoder_layers(model)
        self.recipe = tuple(recipe)
        validate_recipe(self.layers, self.recipe)
        self.entries = []
        self.calls = {}
        for item in self.recipe:
            norm = float(self.layers[item.layer].mlp.down_proj.weight[:, item.neuron]
                         .detach().float().norm().cpu())
            median = medians[item.layer]
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError(f"Invalid down_proj column norm: {norm}.")
            if not math.isfinite(median) or median <= 0:
                raise ValueError(f"Invalid calibration median: {median}.")
            self.entries.append({**asdict(item), "residual_median": median,
                                 "column_norm": norm, "offset": item.dose * median / norm})

    @contextmanager
    def applied(self):
        """Add offsets to the gated MLP activation before down_proj, all tokens.

        Applies to prompt prefill AND every cached decoding step. Scales are
        frozen from the clean prompt, so single-token decode needs no median.
        """
        grouped, handles = {}, []
        self.calls = {}
        for entry in self.entries:
            if entry["dose"] != 0:
                grouped.setdefault(entry["layer"], []).append(entry)

        def make_hook(index, entries):
            def hook(module, args):
                activation = args[0].clone()
                for entry in entries:
                    activation[..., entry["neuron"]] += entry["offset"]
                self.calls[index] = self.calls.get(index, 0) + 1
                return (activation, *args[1:])
            return hook

        try:
            for index, entries in grouped.items():
                handles.append(self.layers[index].mlp.down_proj.register_forward_pre_hook(
                    make_hook(index, entries)))
            yield self
        finally:
            for handle in handles:
                handle.remove()


def select_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_dtype(device, requested="auto"):
    if requested != "auto":
        return getattr(torch, requested)
    if device == "cpu":
        return torch.float32
    if device.startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16
