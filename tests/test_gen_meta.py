"""The generation metadata: it must describe the image, not the intention.

Two holes, of the same kind -- an image that cannot be reproduced from its own
file, or worse, that asserts something false:

1. THE LIST WAS THE ONE OF THE LoRAs ASKED FOR, not of the LoRAs applied (LORAS, not
   _APPLIED_LORAS). A LoRA can be discarded along the way -- a missing file, a refused
   format -- and signing an image with a LoRA it does not carry is a quiet
   lie.
2. THE BASE REPO was missing as soon as a single-file was chosen. A single-file only
   replaces the transformer: the VAE, the text encoder and the architecture config
   come from the repo, so `model` alone reproduces nothing.

Ported from crispz-klein 1.31.0.

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
    assert "base_repo" not in m, "redundant when the model IS the repo"
    print("OK test_the_base_repo_alone_needs_no_second_line")


def test_the_a1111_chunk_carries_them_too():
    """That is the line Civitai and the A1111 viewers read."""
    old = _state(ZIMAGE_TRANSFORMER=CKPT,
                 _APPLIED_LORAS=[("/l/style.safetensors", 0.8)])
    try:
        line = IO._a1111_parameters(P._gen_meta("txt2img", "p", seed=1, steps=8))
    finally:
        _restore(old)
    assert "Loras: style.safetensors@0.8" in line, line
    assert "Base: " in line, line
    print("OK test_the_a1111_chunk_carries_them_too")


# ---------------------------------------------------------------------------
# The INPUT image (ported from crispz-klein 1.32.0). Only one of the four outputs
# named it: the batch (a hardcoded basename), not the plain img2img, not the inpaint, and
# the edit only wrote the NUMBER of references. The default name, not the path: the PNG
# travels, and Gradio drops the uploads into a temporary folder where only the base name
# carries the original name.
# ---------------------------------------------------------------------------

# Full input paths: os.path.join keeps the separator of the running OS, and
# os.path.basename does not split on a backslash under Linux, where the CI runs.
TMP_UPLOAD = os.path.join("C:" + os.sep, "Users", "x", "AppData", "Local", "Temp",
                          "gradio", "ab12", "ma_photo.png")
SRC = os.path.join("F:" + os.sep, "in", "shot.png")
OTHER = os.path.join("F:" + os.sep, "in", "other.png")


def _pil(name=None):
    from PIL import Image
    im = Image.new("RGB", (8, 8))
    if name:
        im.filename = name
    return im


def test_a_path_a_pil_and_an_editor_all_give_the_name():
    assert P.source_meta(SRC) == {"source": "shot.png"}
    assert P.source_meta(_pil(TMP_UPLOAD)) == {"source": "ma_photo.png"}
    assert P.source_meta({"background": _pil(TMP_UPLOAD),
                          "composite": _pil()}) == {"source": "ma_photo.png"}
    print("OK test_a_path_a_pil_and_an_editor_all_give_the_name")


def test_an_unknown_source_records_nothing():
    assert P.source_meta(_pil()) == {}
    assert P.source_meta(None) == {}
    assert P.source_meta([None, None]) == {}
    print("OK test_an_unknown_source_records_nothing")


def test_several_references_come_back_as_a_list():
    got = P.source_meta([_pil(TMP_UPLOAD), None, OTHER, None], "ref_images")
    assert got == {"ref_images": ["ma_photo.png", "other.png"]}, got
    print("OK test_several_references_come_back_as_a_list")


def test_full_and_off_are_honoured():
    old = P.METADATA_SOURCE
    try:
        P.METADATA_SOURCE = "full"
        got = P.source_meta(SRC)["source"]
        assert got.endswith("shot.png") and "in" in got, got
        P.METADATA_SOURCE = "off"
        assert P.source_meta(SRC) == {}
    finally:
        P.METADATA_SOURCE = old
    print("OK test_full_and_off_are_honoured")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All generation-metadata tests passed.")
