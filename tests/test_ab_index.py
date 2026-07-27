"""Tests du cache de metadonnees de l'Asset Browser (reindexation).

Regression: chaque ouverture relisait les tags PNG de TOUTES les images
(~25 ms/image -> 295 s pour 9278 images), au point que le polling du SPA (180 s)
expirait avant la fin -> "il manque des images".

Run:  .venv/Scripts/python tests/test_ab_index.py
"""
import os
import sys
import json
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402
from PIL.PngImagePlugin import PngInfo  # noqa: E402

import cz_assetbrowser as AB  # noqa: E402


def _outdir(n=5, day="2026-07-27"):
    d = tempfile.mkdtemp()
    sub = os.path.join(d, day)
    os.makedirs(sub)
    for i in range(n):
        # format REEL lu par cz_imageio._read_image_meta: chunk PNG 'crispz' = JSON
        info = PngInfo()
        info.add_text("crispz", json.dumps({"prompt": f"prompt {i}", "steps": 8,
                                            "seed": 1000 + i}))
        Image.new("RGB", (64, 64), (i * 20 % 255, 40, 90)).save(
            os.path.join(sub, f"img{i}.png"), pnginfo=info)
    return d


def _count_reads(monkey):
    """Compte les appels reels a _read_image_meta."""
    calls = []
    real = AB._read_image_meta

    def counting(p):
        calls.append(p)
        return real(p)
    AB._read_image_meta = counting
    return calls, real


def test_second_pass_uses_cache():
    d = _outdir(6)
    calls, real = _count_reads(None)
    try:
        AB.ab_reindex(d, gen_thumbs=False)
        first = len(calls)
        calls.clear()
        AB.ab_reindex(d, gen_thumbs=False)
        second = len(calls)
    finally:
        AB._read_image_meta = real
    assert first == 6, f"1re passe doit lire les 6 images, lu {first}"
    assert second == 0, f"2e passe doit tout prendre au cache, relu {second}"


def test_modified_image_is_reread():
    d = _outdir(4)
    calls, real = _count_reads(None)
    try:
        AB.ab_reindex(d, gen_thumbs=False)
        calls.clear()
        # on modifie UNE image -> elle seule doit etre relue
        target = os.path.join(d, "2026-07-27", "img2.png")
        time.sleep(1.1)                      # mtime a la seconde
        info = PngInfo()
        info.add_text("crispz", json.dumps({"prompt": "nouveau prompt", "seed": 999}))
        Image.new("RGB", (64, 64), (7, 7, 7)).save(target, pnginfo=info)
        AB.ab_reindex(d, gen_thumbs=False)
    finally:
        AB._read_image_meta = real
    assert len(calls) == 1, f"seule l'image modifiee doit etre relue, {len(calls)} lues"
    assert calls[0].endswith("img2.png")


def test_cache_reflects_new_metadata():
    d = _outdir(3)
    AB.ab_reindex(d, gen_thumbs=False)
    target = os.path.join(d, "2026-07-27", "img1.png")
    time.sleep(1.1)
    info = PngInfo()
    info.add_text("crispz", json.dumps({"prompt": "un chat roux", "steps": 20,
                                        "seed": 4242}))
    Image.new("RGB", (64, 64), (9, 9, 9)).save(target, pnginfo=info)
    AB.ab_reindex(d, gen_thumbs=False)
    man = json.load(open(os.path.join(d, "_index", "manifest.json"), encoding="utf-8"))
    e = next(x for x in man["images"] if x["file"].endswith("img1.png"))
    assert "chat roux" in (e.get("prompt") or ""), e.get("prompt")
    assert str(e.get("seed")) == "4242", e.get("seed")


def test_deleted_images_leave_the_cache():
    d = _outdir(4)
    AB.ab_reindex(d, gen_thumbs=False)
    os.remove(os.path.join(d, "2026-07-27", "img0.png"))
    AB.ab_reindex(d, gen_thumbs=False)
    cache = json.load(open(os.path.join(d, "_index", "meta_cache.json"), encoding="utf-8"))
    files = cache["files"]
    assert len(files) == 3, f"le cache doit suivre les suppressions, {len(files)} entrees"
    assert not any(k.endswith("img0.png") for k in files)


def test_corrupt_cache_is_ignored_not_fatal():
    d = _outdir(3)
    AB.ab_reindex(d, gen_thumbs=False)
    p = os.path.join(d, "_index", "meta_cache.json")
    with open(p, "w", encoding="utf-8") as f:
        f.write("{ ceci n'est pas du json")
    n, _idx, _j = AB.ab_reindex(d, gen_thumbs=False)   # ne doit pas lever
    assert n == 3
    json.load(open(p, encoding="utf-8"))               # reecrit valide


if __name__ == "__main__":
    for fn in (test_second_pass_uses_cache, test_modified_image_is_reread,
               test_cache_reflects_new_metadata, test_deleted_images_leave_the_cache,
               test_corrupt_cache_is_ignored_not_fatal):
        fn()
        print(f"OK {fn.__name__}")
    print("All asset-browser index tests passed.")
