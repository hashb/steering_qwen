"""Offline correctness tests, including a tiny real Qwen3.5 hybrid decoder."""

from types import SimpleNamespace
import unittest

import torch
from torch import nn
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from steering import Neuron, Steering, calibrate, decoder_layers, validate_recipe


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.mlp.down_proj.weight.copy_(torch.tensor([[3., 0., 1.], [4., 2., 0.]]))

    def forward(self, x):
        return self.mlp.down_proj(x)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ToyBlock()])

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.model.layers[0](input_ids.float()))


class SteeringTests(unittest.TestCase):
    def test_exact_scaling_and_addition_at_every_token(self):
        model = ToyModel()
        recipe = [Neuron(0, 0, 1.), Neuron(0, 1, -1.)]
        # Column norms are 5 and 2; median 10 -> offsets +2 and -5.
        intervention = Steering(model, recipe, {0: 10.})
        self.assertEqual([e["offset"] for e in intervention.entries], [2., -5.])
        x = torch.randn(1, 4, 3)
        weights = model.model.layers[0].mlp.down_proj.weight.detach().clone()
        clean = model(input_ids=x).logits
        with intervention.applied():
            changed = model(input_ids=x).logits
        # W[:,0] * 2 + W[:,1] * -5 == [6,-2].
        expected = torch.tensor([6., -2.]).expand_as(clean)
        torch.testing.assert_close(changed - clean, expected)
        torch.testing.assert_close(model(input_ids=x).logits, clean, rtol=0, atol=0)
        torch.testing.assert_close(model.model.layers[0].mlp.down_proj.weight, weights, rtol=0, atol=0)

    def test_calibration_excludes_first_token_and_averages_middle_pair(self):
        # Output norms are [500,5,15]; tokens[1:] median is 10.
        # Calibration API expects 2D token IDs. This mock maps IDs to activation.
        class TokenModel(ToyModel):
            def forward(self, input_ids, **kwargs):
                x = torch.stack((input_ids.float(), torch.zeros_like(input_ids), torch.zeros_like(input_ids)), -1)
                return super().forward(x)
        model = TokenModel()
        medians, _ = calibrate(model, {"input_ids": torch.tensor([[100, 1, 3]])}, [Neuron(0, 0, 1.)])
        self.assertEqual(medians, {0: 10.})

    def test_cleanup_after_error_and_invalid_indices(self):
        model = ToyModel()
        proj = model.model.layers[0].mlp.down_proj
        intervention = Steering(model, [Neuron(0, 0, 1.)], {0: 10.})
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with intervention.applied():
                raise RuntimeError("intentional")
        self.assertEqual(len(proj._forward_pre_hooks), 0)
        for recipe in ([Neuron(-1, 0, 1.)], [Neuron(0, 3, 1.)], [Neuron(0, 0, float("nan"))]):
            with self.assertRaises(ValueError):
                validate_recipe(decoder_layers(model), recipe)

    def test_tiny_qwen_hybrid_prefill_cached_decode_and_zero_dose(self):
        torch.manual_seed(42)
        config = Qwen3_5TextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_key_head_dim=16, linear_value_head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=2,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                             "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]},
            eos_token_id=None, pad_token_id=0, tie_word_embeddings=True)
        config._attn_implementation = "eager"
        model = Qwen3_5ForCausalLM(config).eval()
        inputs = {"input_ids": torch.tensor([[3, 8, 12, 7]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
        recipe = [Neuron(1, 9, 0.5), Neuron(2, 7, -0.5)]
        with torch.inference_mode():
            medians, clean = calibrate(model, inputs, recipe)
            weights = {name: p.clone() for name, p in model.named_parameters()}
            zero = Steering(model, [Neuron(n.layer, n.neuron, 0.) for n in recipe], medians)
            with zero.applied():
                unchanged = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu()
            torch.testing.assert_close(unchanged, clean, atol=0, rtol=0)
            steering = Steering(model, recipe, medians)
            with steering.applied():
                changed = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1]
                prefill = model(**inputs, use_cache=True, logits_to_keep=1)
                decoded = model(input_ids=torch.tensor([[5]]), past_key_values=prefill.past_key_values,
                                use_cache=True, logits_to_keep=1).logits
                full = model(input_ids=torch.tensor([[3, 8, 12, 7, 5]]), use_cache=False, logits_to_keep=1).logits
                torch.testing.assert_close(decoded, full, atol=2e-5, rtol=2e-4)
            self.assertFalse(torch.allclose(changed.cpu(), clean))
            self.assertEqual(steering.calls, {1: 4, 2: 4})
            with Steering(model, recipe, medians).applied() as generation_steering:
                output = model.generate(**inputs, max_new_tokens=3, do_sample=False, use_cache=True)
            self.assertEqual(output.shape, (1, 7))
            self.assertEqual(generation_steering.calls, {1: 3, 2: 3})
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, weights[name], atol=0, rtol=0)
            for layer in decoder_layers(model):
                self.assertEqual(len(layer.mlp.down_proj._forward_pre_hooks), 0)
                self.assertEqual(len(layer._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
