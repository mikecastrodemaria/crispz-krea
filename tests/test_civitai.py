"""Unit tests for the local/pure CivitAI helpers (no network hit).

Run:  .venv/Scripts/python tests/test_civitai.py
"""
import os
import sys
import json
import hashlib
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_civitai  # noqa: E402


def _tmpfile(data=b"hello crispz"):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "model.safetensors")
    with open(p, "wb") as f:
        f.write(data)
    return p


def test_compute_sha256_progress():
    p = _tmpfile(b"x" * (3 << 20))          # 3 MB -> plusieurs chunks de 1 MB
    events = []
    h = cz_civitai._compute_sha256(p, progress=lambda ph, fr, tx: events.append((ph, fr)))
    assert h == hashlib.sha256(b"x" * (3 << 20)).hexdigest()
    assert events and all(ph == "hash" for ph, _ in events)
    fracs = [fr for _, fr in events]
    assert fracs == sorted(fracs) and abs(fracs[-1] - 1.0) < 1e-9     # croissant -> 100%


def test_model_sha256_reads_sidecar_no_hash():
    p = _tmpfile(b"anything")
    sha = "a" * 64
    with open(os.path.splitext(p)[0] + ".metadata.json", "w", encoding="utf-8") as f:
        json.dump({"sha256": sha.upper()}, f)
    # doit lire le sidecar (minuscule) sans hasher le fichier -> progress jamais appele
    called = []
    got = cz_civitai.model_sha256(p, progress=lambda *a: called.append(a))
    assert got == sha and not called


def test_fetch_missing_file_no_network():
    res = cz_civitai.fetch_civitai_for_model("D:/nope/does-not-exist.safetensors")
    assert res["success"] is False and "not found" in res["message"]


def test_has_preview_and_sidecar_load():
    p = _tmpfile()
    assert cz_civitai.has_preview(p) is False
    assert cz_civitai.load_civitai_sidecar(p) == {}
    stem = os.path.splitext(p)[0]
    with open(stem + ".preview.png", "wb") as f:
        f.write(b"\x89PNG")
    with open(stem + ".civitai.json", "w", encoding="utf-8") as f:
        json.dump({"trainedWords": ["foo"], "examples": [{"url": "u", "prompt": "p"}]}, f)
    assert cz_civitai.has_preview(p) is True
    sc = cz_civitai.load_civitai_sidecar(p)
    assert sc["trainedWords"] == ["foo"] and sc["examples"][0]["prompt"] == "p"


if __name__ == "__main__":
    for fn in (test_compute_sha256_progress, test_model_sha256_reads_sidecar_no_hash,
               test_fetch_missing_file_no_network, test_has_preview_and_sidecar_load):
        fn()
        print(f"OK {fn.__name__}")
    print("All civitai tests passed.")
