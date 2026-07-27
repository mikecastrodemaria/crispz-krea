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


# --------------------------------------------------------------------------
# Index par jour + hook incremental (architecture Fooocus)
# --------------------------------------------------------------------------

def _enable_ab():
    import cz_core
    cz_core.CONFIG.setdefault("asset_browser", {})["enabled"] = True


def test_per_day_manifests_and_days_index():
    d = tempfile.mkdtemp()
    for day, n in (("2026-07-25", 2), ("2026-07-26", 3)):
        sub = os.path.join(d, day)
        os.makedirs(sub)
        for i in range(n):
            info = PngInfo()
            info.add_text("crispz", json.dumps({"prompt": f"{day} {i}"}))
            Image.new("RGB", (32, 32)).save(os.path.join(sub, f"i{i}.png"), pnginfo=info)
    AB.ab_reindex(d, gen_thumbs=False)
    idx = json.load(open(os.path.join(d, "_index", "days.json"), encoding="utf-8"))
    assert idx["total"] == 5
    assert [x["date"] for x in idx["days"]] == ["2026-07-26", "2026-07-25"]  # recent en tete
    assert [x["count"] for x in idx["days"]] == [3, 2]
    # chaque jour a son propre manifest, dans son dossier
    m = json.load(open(os.path.join(d, "2026-07-26", "manifest.json"), encoding="utf-8"))
    assert m["count"] == 3 and all(e["day"] == "2026-07-26" for e in m["images"])
    # days.json doit rester minuscule devant le manifest global
    assert os.path.getsize(os.path.join(d, "_index", "days.json")) < \
        os.path.getsize(os.path.join(d, "_index", "manifest.json"))


def test_incremental_hook_adds_without_rescan():
    _enable_ab()
    d = tempfile.mkdtemp()
    sub = os.path.join(d, "2026-07-27")
    os.makedirs(sub)
    Image.new("RGB", (32, 32)).save(os.path.join(sub, "a.png"))
    AB.ab_reindex(d, gen_thumbs=False)
    p = os.path.join(sub, "b.png")
    Image.new("RGB", (32, 32)).save(p)
    assert AB.on_image_saved(p, output_dir=d, meta={"prompt": "phare", "seed": 7}) is True
    man = json.load(open(os.path.join(sub, "manifest.json"), encoding="utf-8"))
    assert man["count"] == 2
    assert man["images"][0]["file"].endswith("b.png")        # plus recent en tete
    assert man["images"][0]["prompt"] == "phare"
    idx = json.load(open(os.path.join(d, "_index", "days.json"), encoding="utf-8"))
    assert idx["total"] == 2


def test_incremental_hook_is_idempotent():
    _enable_ab()
    d = tempfile.mkdtemp()
    sub = os.path.join(d, "2026-07-27")
    os.makedirs(sub)
    p = os.path.join(sub, "a.png")
    Image.new("RGB", (32, 32)).save(p)
    for _ in range(3):
        AB.on_image_saved(p, output_dir=d, meta={"prompt": "x"})
    man = json.load(open(os.path.join(sub, "manifest.json"), encoding="utf-8"))
    assert man["count"] == 1, f"pas de doublon attendu, {man['count']} entrees"


def test_incremental_hook_never_raises():
    _enable_ab()
    d = tempfile.mkdtemp()
    # fichier hors du dossier de sortie, fichier inexistant, non-image: tout doit
    # renvoyer False sans lever (une generation ne doit jamais casser la-dessus).
    assert AB.on_image_saved(os.path.join(tempfile.mkdtemp(), "ailleurs.png"),
                             output_dir=d) is False
    assert AB.on_image_saved(os.path.join(d, "absent.png"), output_dir=d) is False
    txt = os.path.join(d, "note.txt")
    open(txt, "w").write("x")
    assert AB.on_image_saved(txt, output_dir=d) is False


def test_reindex_and_hook_produce_the_same_entry_shape():
    """Les deux chemins passent par _entry_for -> memes cles, pas de divergence."""
    _enable_ab()
    d = tempfile.mkdtemp()
    sub = os.path.join(d, "2026-07-27")
    os.makedirs(sub)
    info = PngInfo()
    info.add_text("crispz", json.dumps({"prompt": "p", "seed": 1, "steps": 8}))
    Image.new("RGB", (32, 32)).save(os.path.join(sub, "a.png"), pnginfo=info)
    AB.ab_reindex(d, gen_thumbs=False)
    from_reindex = json.load(open(os.path.join(sub, "manifest.json"),
                                  encoding="utf-8"))["images"][0]
    p = os.path.join(sub, "b.png")
    Image.new("RGB", (32, 32)).save(p, pnginfo=info)
    AB.on_image_saved(p, output_dir=d, meta={"prompt": "p", "seed": 1, "steps": 8})
    from_hook = json.load(open(os.path.join(sub, "manifest.json"),
                               encoding="utf-8"))["images"][0]
    assert set(from_reindex) == set(from_hook), \
        f"cles differentes: {set(from_reindex) ^ set(from_hook)}"


if __name__ == "__main__":
    for fn in (test_second_pass_uses_cache, test_modified_image_is_reread,
               test_cache_reflects_new_metadata, test_deleted_images_leave_the_cache,
               test_corrupt_cache_is_ignored_not_fatal,
               test_per_day_manifests_and_days_index,
               test_incremental_hook_adds_without_rescan,
               test_incremental_hook_is_idempotent,
               test_incremental_hook_never_raises,
               test_reindex_and_hook_produce_the_same_entry_shape):
        fn()
        print(f"OK {fn.__name__}")
    print("All asset-browser index tests passed.")
