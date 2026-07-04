"""Unit tests for the job queue pure helpers (no Gradio event needed).

Run:  .venv/Scripts/python tests/test_queue.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_ui  # noqa: E402


def _stub_vals(prompt="a cat", use_input=False, w=1024, h=768, steps=8, n=2, seed=42):
    """36-slot stand-in for _gen_inputs values, with the indexed slots filled."""
    vals = [None] * 36
    vals[cz_ui._Q_IDX["prompt"]] = prompt
    vals[cz_ui._Q_IDX["use_input"]] = use_input
    vals[cz_ui._Q_IDX["width"]] = w
    vals[cz_ui._Q_IDX["height"]] = h
    vals[cz_ui._Q_IDX["gen_steps"]] = steps
    vals[cz_ui._Q_IDX["image_number"]] = n
    vals[cz_ui._Q_IDX["seed"]] = seed
    return vals


def test_label():
    ms = {"base_repo": "Tongyi-MAI/Z-Image-Turbo", "transformer": None}
    lbl = cz_ui._q_label(_stub_vals(), ms)
    assert "txt2img" in lbl and "Z-Image-Turbo" in lbl and "1024x768" in lbl
    assert "8 steps" in lbl and "seed 42" in lbl and "x2" in lbl and "a cat" in lbl
    # transformer wins over base repo; img2img mode; long prompt truncated
    ms2 = {"base_repo": "x", "transformer": "D:/models/juggernaut_z.safetensors"}
    lbl2 = cz_ui._q_label(_stub_vals(prompt="p" * 80, use_input=True), ms2)
    assert "img2img" in lbl2 and "juggernaut_z.safetensors" in lbl2 and "…" in lbl2


def test_move():
    items = [{"label": "a"}, {"label": "b"}, {"label": "c"}]
    out, sel = cz_ui._q_move(items, 2, -1)
    assert [i["label"] for i in out] == ["a", "c", "b"] and sel == 1
    out, sel = cz_ui._q_move(items, 0, -1)          # bord haut: inchange
    assert [i["label"] for i in out] == ["a", "b", "c"] and sel == 0
    out, sel = cz_ui._q_move(items, None, 1)         # pas de selection
    assert sel is None and len(out) == 3
    assert items == [{"label": "a"}, {"label": "b"}, {"label": "c"}]  # pure (copie)


def test_remove():
    items = [{"label": "a"}, {"label": "b"}, {"label": "c"}]
    out, sel = cz_ui._q_remove(items, 1)
    assert [i["label"] for i in out] == ["a", "c"] and sel == 1
    out, sel = cz_ui._q_remove(out, 1)
    assert [i["label"] for i in out] == ["a"] and sel == 0
    out, sel = cz_ui._q_remove(out, 0)
    assert out == [] and sel is None
    out, sel = cz_ui._q_remove([], None)
    assert out == [] and sel is None


def test_render():
    upd, md, btn = cz_ui._q_render([])
    assert "empty" in md and btn["value"] == "+ Queue (0)"
    items = [{"label": "j1"}, {"label": "j2"}]
    upd, md, btn = cz_ui._q_render(items, 1)
    assert "1. j1" in md and "2. j2" in md
    assert btn["value"] == "+ Queue (2)" and upd["value"] == 1
    upd, _, _ = cz_ui._q_render(items, 99)           # selection hors bornes -> None
    assert upd["value"] is None


def test_model_state_roundtrip_keys():
    ms = cz_ui._q_model_state()
    assert set(ms) == {"base_repo", "transformer", "loras", "sampler", "schedule"}


if __name__ == "__main__":
    for fn in (test_label, test_move, test_remove, test_render,
               test_model_state_roundtrip_keys):
        fn()
        print(f"OK {fn.__name__}")
    print("All queue tests passed.")
