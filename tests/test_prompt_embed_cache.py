"""Encoder un prompt deplace l'encodeur Qwen3 sur le GPU. Le faire deux fois pour
le meme texte est du transfert pur.

En offload 'model' ce cout est paye a CHAQUE appel de pipeline. Le detailer le
paie une fois par main, avec le MEME prompt (vide par defaut). Mesure sur
klein-9B GGUF, passe de main a 4 steps: prompt+setup 5,8 s pour 0,3 s de
diffusion -- le calcul avait disparu, restait le deplacement des poids.

encode_prompt() court-circuite l'encodeur des qu'on lui passe prompt_embeds.
Aucun modele n'est charge ici: le pipeline est un faux.

Run:  .venv/Scripts/python tests/test_prompt_embed_cache.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import cz_pipeline as P


class FakePipe:
    """Compte les encodages et retient ce que __call__ a recu."""

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
        assert kw["prompt"] is None, "le prompt doit laisser la place aux embeddings"
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
    """Un embedding calcule par un AUTRE encodeur serait faux."""
    _fresh()
    pipe = FakePipe()
    P._flux_call(pipe, prompt="a lighthouse")
    P.free_vram()
    assert not P._EMBED_CACHE, "le cache doit tomber avec le pipeline"
    P._flux_call(pipe, prompt="a lighthouse")
    assert pipe.encodes == 2, pipe.encodes
    print("OK test_freeing_vram_clears_the_cache")


def test_an_encoder_failure_never_breaks_the_render():
    """Regle maison: un cache ne doit jamais couter un rendu."""
    _fresh()
    pipe = FakePipe(boom=True)
    out = P._flux_call(pipe, prompt="a lighthouse")
    assert out.images == ["IMG"], "le rendu doit aboutir quand meme"
    assert pipe.calls[0]["prompt"] == "a lighthouse", "repli sur le prompt en clair"
    assert "prompt_embeds" not in pipe.calls[0]
    print("OK test_an_encoder_failure_never_breaks_the_render")


def test_disabled_by_config():
    _fresh(maxsize=0)
    pipe = FakePipe()
    P._flux_call(pipe, prompt="a lighthouse")
    P._flux_call(pipe, prompt="a lighthouse")
    assert pipe.encodes == 0, "desactive: aucun encodage anticipe"
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
    """Un appelant qui fournit deja ses embeddings n'est pas contredit."""
    _fresh()
    pipe = FakePipe()
    mine = torch.ones(1, 4, 8)
    P._flux_call(pipe, prompt="a lighthouse", prompt_embeds=mine)
    assert pipe.encodes == 0
    assert pipe.calls[0]["prompt_embeds"] is mine
    print("OK test_an_explicit_prompt_embeds_wins")




def test_every_returned_tensor_is_carried():
    """encode_prompt ne renvoie pas qu'un tenseur selon la famille (masque chez
    Qwen/Krea2, pooled chez Flux). Les oublier ferait planter __call__."""
    _fresh()
    pipe = FakePipe()
    P._flux_call(pipe, prompt="a lighthouse")
    for name in P._EMBED_OUTS:
        assert name in pipe.calls[0], f"{name} manquant dans l'appel"
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
