"""Unit tests for the LoRA pre-flight checks (foreign architecture, FP8).

Two things used to break a render with a wall of diffusers output:

  - a Stable Diffusion / SDXL LoRA dropped in the Flux folder died on
    "Incompatible keys detected:" followed by two thousand key names, which the UI
    reported as a failed hot-swap and paid for with a full model reload, on every render
    since the slot stayed selected. 38 files of 406 in the real folder.
  - an FP8 LoRA died on `"mul_cpu_reduced_float" not implemented for 'Float8_e4m3fn'`:
    diffusers scales the weights by alpha/rank on CPU before the load.

No model is loaded: the state dicts are synthetic and tiny.

Run:  .venv/Scripts/python tests/test_lora_arch.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import cz_pipeline as P  # noqa: E402

DIM, RANK = 8, 2


def _pair(sd, name):
    sd[f"{name}.lora_down.weight"] = torch.zeros(RANK, DIM)
    sd[f"{name}.lora_up.weight"] = torch.zeros(DIM, RANK)
    sd[f"{name}.alpha"] = torch.tensor(float(RANK))


def _sdxl_sd(style="original"):
    """A kohya SDXL LoRA: an SD UNet plus the CLIP text encoder."""
    sd = {}
    if style == "original":                       # original / kohya naming
        _pair(sd, "lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q")
        _pair(sd, "lora_unet_middle_block_1_transformer_blocks_0_attn1_to_k")
        _pair(sd, "lora_unet_output_blocks_3_1_transformer_blocks_0_attn2_to_v")
    else:                                         # diffusers naming
        _pair(sd, "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q")
        _pair(sd, "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_k")
        _pair(sd, "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_v")
    _pair(sd, "lora_te_text_model_encoder_layers_0_mlp_fc1")
    return sd


def _flux_sd():
    sd = {}
    _pair(sd, "lora_unet_double_blocks_0_img_attn_proj")
    _pair(sd, "lora_unet_single_blocks_0_linear1")
    return sd


def _write(sd, name="l.safetensors"):
    d = tempfile.mkdtemp()
    p = os.path.join(d, name)
    save_file(sd, p)
    return p


def test_sdxl_lora_is_refused_with_a_reason():
    for style in ("original", "diffusers"):
        why = P.foreign_lora_reason(_write(_sdxl_sd(style)))
        assert why, style
        assert "SDXL" in why and "FLUX" in why, why
        # the reason must NAME what it saw, not just say no
        assert any(m in why for m in ("input_blocks", "down_blocks")), why


def test_the_reason_counts_the_text_encoder_keys():
    why = P.foreign_lora_reason(_write(_sdxl_sd()))
    assert "CLIP text-encoder keys" in why, why


def test_a_flux_lora_passes():
    assert P.foreign_lora_reason(_write(_flux_sd())) == ""


def test_an_unreadable_file_is_left_to_diffusers():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "broken.safetensors")
    with open(p, "wb") as f:
        f.write(b"garbage")
    assert P.foreign_lora_reason(p) == ""          # must not raise, must not accuse


def test_a_healthy_lora_keeps_the_tested_route():
    """The folder + weight_name route is the one that works offline: a file that needs
    nothing must not be turned into a dict."""
    p = _write(_flux_sd(), "healthy.safetensors")
    src, kw = P._lora_source(p)
    assert src == os.path.dirname(p) and kw == {"weight_name": "healthy.safetensors"}


def test_an_fp8_lora_is_upcast():
    sd = {k: (v.to(torch.float8_e4m3fn) if v.dtype.is_floating_point else v)
          for k, v in _flux_sd().items()}
    src, kw = P._lora_source(_write(sd, "fp8.safetensors"))
    assert isinstance(src, dict) and kw == {}
    assert not [t for t in src.values()
                if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)]
    assert all(t.dtype == P.DTYPE for t in src.values() if t.dim() > 0)


def test_meta_params_detects_a_meta_module():
    assert P._meta_params(torch.nn.Linear(4, 4)) == []
    with torch.device("meta"):
        ghost = torch.nn.Linear(4, 4)
    assert "weight" in P._meta_params(ghost)
    assert P._meta_params(None) == []


if __name__ == "__main__":
    for fn in (test_sdxl_lora_is_refused_with_a_reason,
               test_the_reason_counts_the_text_encoder_keys,
               test_a_flux_lora_passes,
               test_an_unreadable_file_is_left_to_diffusers,
               test_a_healthy_lora_keeps_the_tested_route,
               test_an_fp8_lora_is_upcast,
               test_meta_params_detects_a_meta_module):
        fn()
        print(f"OK {fn.__name__}")
    print("All LoRA architecture tests passed.")
