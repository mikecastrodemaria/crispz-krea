"""Metadonnees de generation: elles doivent decrire l'image, pas l'intention.

Deux trous, du meme genre -- une image qu'on ne peut pas reproduire depuis son propre
fichier, ou pire, qui affirme quelque chose de faux:

1. LA LISTE ETAIT CELLE DES LoRA DEMANDEES, pas des LoRA posees (LORAS, pas
   _APPLIED_LORAS). Une LoRA peut etre ecartee en route -- fichier absent, format
   refuse -- et signer une image avec une LoRA qu'elle ne porte pas est un mensonge
   tranquille.
2. LE REPO DE BASE manquait des qu'un single-file etait choisi. Un single-file ne
   remplace que le transformer: le VAE, l'encodeur texte et la config d'architecture
   viennent du repo, donc `model` seul ne reproduit rien.

Porte depuis crispz-klein 1.31.0.

Run:  .venv/Scripts/python tests/test_gen_meta.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_imageio as IO
import cz_pipeline as P

CKPT = r"F:\models\un_checkpoint.safetensors"
STATE = ("BASE_REPO", "ZIMAGE_TRANSFORMER", "LORAS", "_APPLIED_LORAS")


def _state(**kw):
    old = {k: getattr(P, k) for k in STATE}
    P.ZIMAGE_TRANSFORMER = None
    P.LORAS = P._APPLIED_LORAS = []
    for k, v in kw.items():
        setattr(P, k, v)
    return old


def _restore(old):
    for k, v in old.items():
        setattr(P, k, v)


def test_a_refused_lora_is_not_claimed_as_applied():
    old = _state(LORAS=[("/l/ok.safetensors", 0.8), ("/l/gone.safetensors", 0.5)],
                 _APPLIED_LORAS=[("/l/ok.safetensors", 0.8)])
    try:
        m = P._gen_meta("txt2img", "p")
    finally:
        _restore(old)
    assert m["loras"] == ["ok.safetensors@0.8"], m.get("loras")
    assert m["loras_not_applied"] == ["gone.safetensors@0.5"], m.get("loras_not_applied")
    print("OK test_a_refused_lora_is_not_claimed_as_applied")


def test_a_single_file_records_the_base_repo_it_needs():
    old = _state(ZIMAGE_TRANSFORMER=CKPT)
    try:
        m = P._gen_meta("txt2img", "p")
    finally:
        _restore(old)
    assert m["model"] == CKPT, m["model"]
    assert m["base_repo"] == P.BASE_REPO, m.get("base_repo")
    print("OK test_a_single_file_records_the_base_repo_it_needs")


def test_the_base_repo_alone_needs_no_second_line():
    old = _state()
    try:
        m = P._gen_meta("txt2img", "p")
    finally:
        _restore(old)
    assert "base_repo" not in m, "redondant quand le modele EST le repo"
    print("OK test_the_base_repo_alone_needs_no_second_line")


def test_the_a1111_chunk_carries_them_too():
    """C'est la ligne que lisent Civitai et les visionneuses A1111."""
    old = _state(ZIMAGE_TRANSFORMER=CKPT,
                 _APPLIED_LORAS=[("/l/style.safetensors", 0.8)])
    try:
        line = IO._a1111_parameters(P._gen_meta("txt2img", "p", seed=1, steps=8))
    finally:
        _restore(old)
    assert "Loras: style.safetensors@0.8" in line, line
    assert "Base: " in line, line
    print("OK test_the_a1111_chunk_carries_them_too")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All generation-metadata tests passed.")
