"""Unit tests for the quantized-checkpoint support: header routing
(_safetensors_unsupported / _safetensors_dequant), the ComfyUI FP8/INT8 dequant
loader (_load_dequant_state_dict) and the GGUF guards (_gguf_arch /
_gguf_layout_unsupported). Synthetic files only (a few KB), no model download.

Run:  .venv/Scripts/python tests/test_quant_formats.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import cz_pipeline  # noqa: E402

TMP = tempfile.mkdtemp(prefix="cz_quant_test_")


def _st(name, tensors):
    p = os.path.join(TMP, name)
    save_file(tensors, p)
    return p


# Cles minimales "FLUX.1" (marqueur double_blocks) au layout original ComfyUI.
_W = "double_blocks.0.img_attn.proj.weight"


def _pad19(prefix=""):
    """19 double_blocks factices (FLUX.1 en a 19): la garde anti-Klein compte les
    blocks depuis l'en-tete et refuse tout autre nombre."""
    return {f"{prefix}double_blocks.{i}.txt_attn.proj.weight":
            torch.zeros(1, dtype=torch.bfloat16) for i in range(19)}


def test_bf16_passthrough():
    p = _st("bf16.safetensors", {_W: torch.randn(4, 4, dtype=torch.bfloat16)})
    assert cz_pipeline._safetensors_unsupported(p) is None
    assert cz_pipeline._safetensors_dequant(p) is None


def test_fp8_scaled_dequant_math():
    # Format reel de flux1-krea-dev_fp8_scaled: X.weight F8 + X.scale_weight +
    # X.scale_input (jetee) + marqueur global 'scaled_fp8'.
    w = torch.randn(4, 4).to(torch.float8_e4m3fn)
    scale = torch.tensor(2.5, dtype=torch.float32)
    p = _st("fp8s.safetensors", {
        _W: w,
        _W.replace(".weight", ".scale_weight"): scale,
        _W.replace(".weight", ".scale_input"): torch.tensor(1.0),
        "scaled_fp8": torch.zeros(2).to(torch.float8_e4m3fn),
        **_pad19(),
    })
    assert cz_pipeline._safetensors_dequant(p) == "FP8 scaled"
    sd = cz_pipeline._load_dequant_state_dict(p)
    want = (w.to(torch.float32) * 2.5).to(cz_pipeline.DTYPE)
    assert torch.allclose(sd[_W].float(), want.float())
    # metadonnees consommees/jetees, jamais dans le dict final
    assert _W.replace(".weight", ".scale_weight") not in sd
    assert _W.replace(".weight", ".scale_input") not in sd
    assert "scaled_fp8" not in sd


def test_int8_per_row_scale():
    w = torch.randint(-127, 127, (4, 3), dtype=torch.int8)
    scale = torch.rand(4, 1, dtype=torch.float32)
    p = _st("int8.safetensors", {_W: w, _W + "_scale": scale, **_pad19()})
    assert cz_pipeline._safetensors_dequant(p) == "INT8 scaled"
    sd = cz_pipeline._load_dequant_state_dict(p)
    want = (w.to(torch.float32) * scale).to(cz_pipeline.DTYPE)
    assert torch.allclose(sd[_W].float(), want.float())


def test_int8_convrot_roundtrip():
    # Format int8_tensorwise + ConvRot de comfy-quants: les poids sont TOURNES
    # (Hadamard base H4 par groupes de 256) avant quantification -> le loader doit
    # defaire la rotation, sinon bruit total (observe en vrai sur redzit/studio).
    torch.manual_seed(0)
    W = torch.randn(8, 512)                      # in=512 -> 2 groupes de 256
    H = cz_pipeline._hadamard_ortho(256)
    wr = (W.view(8, 2, 256) @ H.T).reshape(8, 512)
    scale = (wr.abs().amax(dim=-1, keepdim=True) / 127).clamp(min=1e-30)
    q = (wr / scale).round().clamp(-128, 127).to(torch.int8)
    blob = torch.tensor(
        list(b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}'),
        dtype=torch.uint8)
    p = _st("convrot.safetensors", {
        _W: q, _W + "_scale": scale.float(),
        _W.replace(".weight", ".comfy_quant"): blob,
        **_pad19(),
    })
    sd = cz_pipeline._load_dequant_state_dict(p)
    err = float((sd[_W].float() - W).abs().max() / W.abs().max())
    assert err < 0.02, f"convrot roundtrip error too high: {err}"


def test_aio_bundle_filtered():
    p = _st("aio.safetensors", {
        "model.diffusion_model." + _W: torch.randn(4, 4, dtype=torch.bfloat16),
        "text_encoders.t5xxl.layers.0.weight": torch.randn(4, 4).to(torch.float8_e4m3fn),
        "vae.decoder.weight": torch.randn(2, 2, dtype=torch.bfloat16),
        **_pad19("model.diffusion_model."),
    })
    assert cz_pipeline._safetensors_dequant(p) == "FP8"
    sd = cz_pipeline._load_dequant_state_dict(p)
    assert "model.diffusion_model." + _W in sd
    assert not any(k.startswith(("text_encoders.", "vae.")) for k in sd)


def test_klein_pruned_rejected():
    # FLUX.2 Klein reutilise les memes cles mais n'a que 5 double_blocks -> refus
    # AVANT de lire les poids (mesure: 3 min de dequant perdues sinon).
    sd = {f"double_blocks.{i}.img_attn.proj.weight":
          torch.randn(2, 2).to(torch.float8_e4m3fn) for i in range(5)}
    p = _st("klein.safetensors", sd)
    raised = False
    try:
        cz_pipeline._load_dequant_state_dict(p)
    except RuntimeError as e:
        raised = "double blocks" in str(e)
    assert raised, "un FLUX.2 Klein / variante pruned doit etre refuse a l'en-tete"


def test_foreign_arch_rejected():
    w = torch.randn(4, 4).to(torch.float8_e4m3fn)
    p = _st("foreign.safetensors",
            {"model.diffusion_model.layers.0.mlp.gate_proj.weight": w})
    raised = False
    try:
        cz_pipeline._load_dequant_state_dict(p)
    except RuntimeError as e:
        raised = "FLUX" in str(e)
    assert raised, "checkpoint quantifie d'une autre archi doit etre refuse clairement"


def test_lora_and_svdq_still_unsupported():
    lora = {f"lora_unet_a{i}.lora_down.weight": torch.zeros(2, 2) for i in range(4)}
    p = _st("lora.safetensors", lora)
    assert "LoRA" in (cz_pipeline._safetensors_unsupported(p) or "")
    p = _st("svdq.safetensors", {"blocks.0.qweight": torch.zeros(2, 2, dtype=torch.int8)})
    assert "SVDQuant" in (cz_pipeline._safetensors_unsupported(p) or "")


def _gguf(name, arch, tensor_name):
    import numpy as np
    from gguf import GGUFWriter
    p = os.path.join(TMP, name)
    w = GGUFWriter(p, arch)
    w.add_tensor(tensor_name, np.zeros((4, 32), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return p


def test_gguf_arch_and_layout():
    ok = _gguf("flux.gguf", "flux", "double_blocks.0.img_attn.qkv.weight")
    assert cz_pipeline._gguf_arch(ok) == "flux"
    assert cz_pipeline._gguf_layout_unsupported(ok) is None
    qwen = _gguf("qwen.gguf", "qwen_image", "transformer_blocks.0.attn.to_q.weight")
    assert cz_pipeline._gguf_arch(qwen) == "qwen_image"
    sdcpp = _gguf("sdcpp.gguf", "flux", "blocks.0.attn.wq.weight")
    assert cz_pipeline._gguf_layout_unsupported(sdcpp) is not None


if __name__ == "__main__":
    for fn in (test_bf16_passthrough, test_fp8_scaled_dequant_math,
               test_int8_per_row_scale, test_int8_convrot_roundtrip,
               test_aio_bundle_filtered, test_klein_pruned_rejected,
               test_foreign_arch_rejected, test_lora_and_svdq_still_unsupported,
               test_gguf_arch_and_layout):
        fn()
        print(f"OK {fn.__name__}")
    print("All quant-format tests passed.")
