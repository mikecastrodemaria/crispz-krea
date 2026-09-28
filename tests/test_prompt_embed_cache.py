"""Encoding a prompt moves the Qwen3 encoder onto the GPU. Doing it twice for
the same text is pure transfer.

In 'model' offload that cost is paid on EVERY pipeline call. The detailer pays
it once per hand, with the SAME prompt (empty by default). Measured on
klein-9B GGUF, a hand pass at 4 steps: prompt+setup 5.8 s for 0.3 s of
diffusion -- the computation had disappeared, what was left was moving the weights.

encode_prompt() short-circuits the encoder as soon as it is passed prompt_embeds.
No model is loaded here: the pipeline is a fake.

Run:  .venv/Scripts/python tests/test_prompt_embed_cache.py

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import cz_pipeline as P


class FakePipe:
    """Counts the encodings and remembers what __call__ received."""

    config = type("c", (), {"is_distilled": True})()
    _execution_device = "cpu"

    def __init__(self, boom=False):
        self.text_encoder = object()
        self.encodes = 0
        self.calls = []
        self.boom = boom

    def encode_prompt(self, prompt=None, device=None, **kw):
        self.encodes += 1
        if self.boom:
            raise RuntimeError("encoder unavailable")
        return tuple(torch.zeros(1, 4, 8) for _ in P._EMBED_OUTS) + (torch.zeros(4, 3),)

    def __call__(self, **kw):
        self.calls.append(kw)
        return type("o", (), {"images": ["IMG"]})()


def _fresh(maxsize=8):
    P._embed_cache_clear()
    P._EMBED_CACHE_MAX = maxsize


def test_same_prompt_encodes_once():
    _fresh()
    pipe = FakePipe()
    for _ in range(3):
        P._flux_call(pipe, prompt="a lighthouse")
    assert pipe.encodes == 1, f"encode appele {pipe.encodes} fois"
    for kw in pipe.calls:
        assert kw["prompt"] is None, "the prompt must give way to the embeddings"
        assert kw["prompt_embeds"] is not None
    print("OK test_same_prompt_encodes_once")


def test_a_different_prompt_is_encoded():
    _fresh()
    pipe = FakePipe()
    P._flux_call(pipe, prompt="a lighthouse")
    P._flux_call(pipe, prompt="a harbour")
    P._flux_call(pipe, prompt="a lighthouse")
    assert pipe.encodes == 2, pipe.encodes
    print("OK test_a_different_prompt_is_encoded")


def test_freeing_vram_clears_the_cache():
    """An embedding computed by ANOTHER encoder would be wrong."""
    _fresh()
    pipe = FakePipe()
    P._flux_call(pipe, prompt="a lighthouse")
    P.free_vram()
    assert not P._EMBED_CACHE, "the cache must go with the pipeline"
    P._flux_call(pipe, prompt="a lighthouse")
    assert pipe.encodes == 2, pipe.encodes
    print("OK test_freeing_vram_clears_the_cache")


def test_an_encoder_failure_never_breaks_the_render():
    """A house rule: a cache must never cost a render."""
    _fresh()
    pipe = FakePipe(boom=True)
    out = P._flux_call(pipe, prompt="a lighthouse")
    assert out.images == ["IMG"], "the render must succeed anyway"
    assert pipe.calls[0]["prompt"] == "a lighthouse", "falls back on the plain prompt"
    assert "prompt_embeds" not in pipe.calls[0]
    print("OK test_an_encoder_failure_never_breaks_the_render")


def test_disabled_by_config():
    _fresh(maxsize=0)
    pipe = FakePipe()
    P._flux_call(pipe, prompt="a lighthouse")
    P._flux_call(pipe, prompt="a lighthouse")
    assert pipe.encodes == 0, "disabled: no early encoding"
    assert pipe.calls[0]["prompt"] == "a lighthouse"
    _fresh()
    print("OK test_disabled_by_config")


def test_the_cache_is_bounded():
    _fresh(maxsize=2)
    pipe = FakePipe()
    for w in ("one", "two", "three"):
        P._flux_call(pipe, prompt=w)
    assert len(P._EMBED_CACHE) == 2, len(P._EMBED_CACHE)
    _fresh()
    print("OK test_the_cache_is_bounded")


def test_an_explicit_prompt_embeds_wins():
    """A caller that already supplies its embeddings is not contradicted."""
    _fresh()
    pipe = FakePipe()
    mine = torch.ones(1, 4, 8)
    P._flux_call(pipe, prompt="a lighthouse", prompt_embeds=mine)
    assert pipe.encodes == 0
    assert pipe.calls[0]["prompt_embeds"] is mine
    print("OK test_an_explicit_prompt_embeds_wins")




def test_every_returned_tensor_is_carried():
    """encode_prompt does not return a single tensor in every family (a mask on
    Qwen/Krea2, a pooled one on Flux). Forgetting them would make __call__ crash."""
    _fresh()
    pipe = FakePipe()
    P._flux_call(pipe, prompt="a lighthouse")
    for name in P._EMBED_OUTS:
        assert name in pipe.calls[0], f"{name} missing from the call"
    print("OK test_every_returned_tensor_is_carried")

if __name__ == "__main__":
    test_same_prompt_encodes_once()
    test_a_different_prompt_is_encoded()
    test_freeing_vram_clears_the_cache()
    test_an_encoder_failure_never_breaks_the_render()
    test_disabled_by_config()
    test_the_cache_is_bounded()
    test_an_explicit_prompt_embeds_wins()
    test_every_returned_tensor_is_carried()
    print("ALL OK")
