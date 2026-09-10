"""Encodeurs texte de remplacement (Models > Checkpoints > Text encoder T5 / CLIP).

FLUX.1 Krea a DEUX encodeurs, chacun son composant diffusers et son choix:
  text_encoder_2 = T5-XXL (T5EncoderModel) -> prompt_embeds, la sequence que lit le
                   transformer (4096 de large, 24 couches);
  text_encoder   = CLIP-L (CLIPTextModel)  -> pooled_prompt_embeds (768, 12 couches).
Un autre encodeur ne se branche que s'il a la meme famille, la meme largeur et le meme
nombre de couches que le MEME composant du repo de base, et le refus doit le dire AVANT
de lire 9 Go de T5.

Ces tests verrouillent aussi ce qui rendrait l'option dangereuse en silence:
  - un changement d'encodeur vide le cache d'embeddings, et les DEUX encodeurs sont dans
    la CLE du cache (elle ne contenait que id(pipe.text_encoder), le CLIP, alors que
    c'est le T5 qui produit prompt_embeds);
  - _ensure_base passe l'encodeur a from_pretrained et, au moindre probleme, retombe
    sur celui du repo sans perdre la generation;
  - les pipes derives (img2img, inpaint: from_pipe) partagent les encodeurs du base;
  - les metadonnees nomment les encodeurs qui ont REELLEMENT tourne, par leur nom de
    dossier et jamais par leur chemin (qui finirait dans les PNG partages);
  - l'UI ne memorise qu'un encodeur valide; la file garde ceux du job.

Aucun modele reel, aucun reseau, aucun GPU: configs factices, pipelines factices ou
minuscules (quelques Ko de poids aleatoires).

Run:  .venv/Scripts/python tests/test_text_encoder.py
"""
import json
import os
import sys
import tempfile

# Jamais de reseau, meme par accident: huggingface_hub lit ceci a l'import.
os.environ["HF_HUB_OFFLINE"] = "1"
# Jamais de GPU non plus: get_pipe deplacerait le pipeline minuscule sur cuda. '-1' et
# non '': sous Windows, une variable vide ne masque pas le GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

import cz_imageio
import cz_pipeline as P

T5, CLIP = "text_encoder_2", "text_encoder"
T5XXL = {"model_type": "t5", "d_model": 4096, "num_layers": 24,
         "architectures": ["T5EncoderModel"]}
CLIPL = {"model_type": "clip_text_model", "hidden_size": 768, "num_hidden_layers": 12,
         "architectures": ["CLIPTextModel"]}
BASE = {T5: T5XXL, CLIP: CLIPL}          # ce que dit black-forest-labs/FLUX.1-Krea-dev
NONE = dict.fromkeys(P.TEXT_ENCODER_COMPONENTS, "")

_TMP = tempfile.TemporaryDirectory(prefix="crispz_te_")   # tout est efface a la sortie


def _mkdtemp(prefix):
    return tempfile.mkdtemp(prefix=prefix, dir=_TMP.name)


def _folder(cfg, sub=None, name="enc", root=None):
    """Dossier d'encodeur factice: un config.json, a la racine ou dans `sub`."""
    d = os.path.join(root or _mkdtemp("te_"), name)
    p = os.path.join(d, sub) if sub else d
    os.makedirs(p, exist_ok=True)
    with open(os.path.join(p, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return d


class _Base:
    """Remplace la lecture des configs d'encodeur du repo de base (pas de reseau, pas de
    HF) et retient quels composants ont ete demandes."""

    def __init__(self, cfgs=BASE):
        self.cfgs, self.asked = cfgs, []

    def __enter__(self):
        self.old = P._base_text_encoder_config

        def fake(base=None, component=T5, local_only=False):
            self.asked.append(component)
            return self.cfgs.get(component)
        P._base_text_encoder_config = fake
        return self

    def __exit__(self, *a):
        P._base_text_encoder_config = self.old


class _Dirs:
    """Les listes ne voient que `te_dir` et `extra`, jamais les vrais dossiers du poste."""

    def __init__(self, te_dir="", extra=""):
        self.new = (te_dir, "", extra)

    def __enter__(self):
        self.old = (P.TEXT_ENCODERS_DIR, P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR)
        P.TEXT_ENCODERS_DIR, P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR = self.new

    def __exit__(self, *a):
        P.TEXT_ENCODERS_DIR, P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR = self.old


def test_same_architecture_is_accepted():
    with _Base():
        assert P._text_encoder_problem(_folder(T5XXL), T5) is None
        # copie d'un repo diffusers: le T5 est dans text_encoder_2/, le CLIP dans text_encoder/
        assert P._text_encoder_problem(_folder(T5XXL, T5), T5) is None
        assert P._text_encoder_problem(_folder(CLIPL, CLIP), CLIP) is None
        assert P._text_encoder_problem(_folder(CLIPL), CLIP) is None
        # T5ForConditionalGeneration complet: T5EncoderModel n'en lit que l'encodeur
        full_t5 = {**T5XXL, "architectures": ["T5ForConditionalGeneration"],
                   "num_decoder_layers": 24}
        assert P._text_encoder_problem(_folder(full_t5), T5) is None
        # CLIPModel complet (texte + vision), forme usuelle d'un CLIP-L fine-tune
        full_clip = {"model_type": "clip", "architectures": ["CLIPModel"],
                     "text_config": {"hidden_size": 768, "num_hidden_layers": 12},
                     "vision_config": {"hidden_size": 1024, "num_hidden_layers": 24}}
        assert P._text_encoder_problem(_folder(full_clip), CLIP) is None
    print("OK test_same_architecture_is_accepted")


def test_each_slot_is_checked_against_its_own_base_component():
    """Le T5 se compare au text_encoder_2 du repo, le CLIP au text_encoder: le meme
    dossier convient a l'un et pas a l'autre."""
    t5 = _folder(T5XXL)
    with _Base() as b:
        assert P._text_encoder_problem(t5, T5) is None
        why = P._text_encoder_problem(t5, CLIP)
    assert why and "'t5'" in why and "clip_text_model" in why, why
    assert b.asked == [T5, CLIP], b.asked
    print("OK test_each_slot_is_checked_against_its_own_base_component")


def test_a_wrong_width_is_refused_with_both_numbers():
    with _Base():
        why = P._text_encoder_problem(_folder({**T5XXL, "d_model": 2048}), T5)   # T5-XL
        assert why and "2048" in why and "4096" in why, why
        why = P._text_encoder_problem(_folder({**CLIPL, "hidden_size": 1280}), CLIP)  # CLIP-G
        assert why and "1280" in why and "768" in why, why
    print("OK test_a_wrong_width_is_refused_with_both_numbers")


def test_other_family_and_layer_count_are_refused():
    with _Base():
        why = P._text_encoder_problem(_folder({**T5XXL, "model_type": "umt5"}), T5)
        assert why and "umt5" in why, why
        why = P._text_encoder_problem(_folder({**T5XXL, "num_layers": 12}), T5)
        assert why and "12" in why and "24" in why, why
    print("OK test_other_family_and_layer_count_are_refused")


def test_gguf_single_file_and_empty_folder_are_refused_with_the_reason():
    with _Base():
        assert "GGUF" in P._text_encoder_problem(r"F:\x\t5-v1_1-xxl-encoder-Q8_0.gguf", T5)
        assert "FOLDER" in P._text_encoder_problem(r"F:\x\t5xxl_fp16.safetensors", T5)
        assert "FOLDER" in P._text_encoder_problem(r"F:\x\clip_l.safetensors", CLIP)
        why = P._text_encoder_problem(_mkdtemp("empty_"), T5)
        assert "config.json" in why and "text_encoder_2/" in why, why
        # un chemin d'une autre machine n'est jamais pris pour un repo HF
        assert "neither" in P._text_encoder_problem("/home/someone/t5xxl", T5)
    print("OK test_gguf_single_file_and_empty_folder_are_refused_with_the_reason")


def test_hf_ids_may_carry_a_subfolder():
    assert P._split_hf_src("owner/repo") == ("owner/repo", None)
    assert P._split_hf_src("owner/repo/text_encoder_2") == ("owner/repo", "text_encoder_2")
    assert P._split_hf_src("owner/repo/a/b") == ("owner/repo", "a/b")
    print("OK test_hf_ids_may_carry_a_subfolder")


def test_the_class_and_the_config_come_from_the_same_base_component():
    base = _mkdtemp("base_")
    with open(os.path.join(base, "model_index.json"), "w", encoding="utf-8") as f:
        json.dump({"_class_name": "FluxPipeline",
                   "text_encoder": ["transformers", "CLIPTextModel"],
                   "text_encoder_2": ["transformers", "T5EncoderModel"]}, f)
    for comp, cfg in BASE.items():
        os.makedirs(os.path.join(base, comp))
        with open(os.path.join(base, comp, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f)
    assert P._encoder_class(base, T5).__name__ == "T5EncoderModel"
    assert P._encoder_class(base, CLIP).__name__ == "CLIPTextModel"
    assert P._base_text_encoder_config(base, T5)["d_model"] == 4096
    assert P._base_text_encoder_config(base, CLIP)["hidden_size"] == 768
    # local_only: un repo absent du cache rend None, sans reseau
    assert P._base_text_encoder_config("nobody/not-a-repo", T5, local_only=True) is None
    print("OK test_the_class_and_the_config_come_from_the_same_base_component")


def test_changing_an_encoder_frees_the_pipe_and_the_cache():
    old = (dict(P.TEXT_ENCODER), P._BASE_PIPE)
    try:
        P.TEXT_ENCODER = dict(NONE)
        P._BASE_PIPE = object()
        P._EMBED_CACHE[("k",)] = ("v",)
        P._TEXT_ENCODER_ACTIVE = {T5: r"D:\enc\old-t5", CLIP: ""}
        P.set_text_encoder(T5, r"D:\enc\t5xxl-abl")
        assert P.TEXT_ENCODER == {T5: r"D:\enc\t5xxl-abl", CLIP: ""}, P.TEXT_ENCODER
        assert P._BASE_PIPE is None, "le pipeline doit etre libere"
        assert not P._EMBED_CACHE, "les anciens encodages resteraient servis"
        assert not any(P._TEXT_ENCODER_ACTIVE.values()), "free_vram garde l'ancien encodeur"
        # meme valeur: rien ne bouge, pas de rechargement inutile
        sentinel = P._BASE_PIPE = object()
        P.set_text_encoder(T5, r"D:\enc\t5xxl-abl")
        assert P._BASE_PIPE is sentinel
        # l'autre composant se change a part, et libere aussi
        P.set_text_encoder(CLIP, r"D:\enc\clip-gmp")
        assert P._BASE_PIPE is None and P.TEXT_ENCODER[T5] == r"D:\enc\t5xxl-abl"
        try:
            P.set_text_encoder("text_encoder_3", "x")
            raise AssertionError("un composant inconnu doit etre refuse")
        except ValueError:
            pass
    finally:
        P.TEXT_ENCODER, P._BASE_PIPE = old
        P._TEXT_ENCODER_ACTIVE = dict(NONE)
        P._EMBED_CACHE.clear()
    print("OK test_changing_an_encoder_frees_the_pipe_and_the_cache")


class FakePipe:
    """Deux encodeurs, comme FluxPipeline; compte les encodages."""

    def __init__(self):
        self.text_encoder = object()        # CLIP
        self.text_encoder_2 = object()      # T5
        self._execution_device = "cpu"
        self.n = 0

    def encode_prompt(self, prompt=None, device=None, **kw):
        self.n += 1
        return (torch.zeros(1, 2, 4), torch.zeros(1, 4), torch.zeros(2, 3))


def test_the_embed_key_carries_both_encoders():
    """Meme prompt, meme pipe: un autre T5 OU un autre CLIP = un autre encodage."""
    P._embed_cache_clear()
    old = dict(P._TEXT_ENCODER_ACTIVE)
    try:
        pipe = FakePipe()
        P._TEXT_ENCODER_ACTIVE = dict(NONE)
        P._cached_prompt_embeds(pipe, "p", {})
        P._cached_prompt_embeds(pipe, "p", {})
        assert pipe.n == 1, pipe.n
        P._TEXT_ENCODER_ACTIVE = {T5: r"D:\enc\t5xxl-abl", CLIP: ""}
        P._cached_prompt_embeds(pipe, "p", {})
        assert pipe.n == 2, "un encodage de l'ancien T5 a ete resservi"
        P._TEXT_ENCODER_ACTIVE = {T5: r"D:\enc\t5xxl-abl", CLIP: r"D:\enc\clip-gmp"}
        P._cached_prompt_embeds(pipe, "p", {})
        assert pipe.n == 3, "un encodage de l'ancien CLIP a ete resservi"
        # le T5 du pipe remplace par un autre objet: l'ancienne cle ne voyait que le CLIP
        keep = pipe.text_encoder_2
        pipe.text_encoder_2 = object()
        P._cached_prompt_embeds(pipe, "p", {})
        assert pipe.n == 4 and keep is not pipe.text_encoder_2, pipe.n
    finally:
        P._TEXT_ENCODER_ACTIVE = old
        P._embed_cache_clear()
    print("OK test_the_embed_key_carries_both_encoders")


def test_metadata_names_the_encoders_that_ran_and_never_their_paths():
    old = (dict(P.TEXT_ENCODER), dict(P._TEXT_ENCODER_ACTIVE))
    t5 = r"C:\Users\someone\models\text_encoders\t5xxl-abliterated"
    clip = r"C:\Users\someone\models\flux-copy\text_encoder"
    try:
        P.TEXT_ENCODER = {T5: t5, CLIP: clip}
        P._TEXT_ENCODER_ACTIVE = {T5: t5, CLIP: clip}
        m = P._gen_meta("txt2img", "p")
        assert m["text_encoder_t5"] == "t5xxl-abliterated", m
        assert m["text_encoder_clip"] == "flux-copy", m
        assert "someone" not in json.dumps(m), "chemin local dans les metadonnees"
        # T5 ecarte au chargement, CLIP applique: chacun son champ
        P._TEXT_ENCODER_ACTIVE = {T5: "", CLIP: clip}
        m = P._gen_meta("txt2img", "p")
        assert "text_encoder_t5" not in m, m
        assert m["text_encoder_t5_not_applied"] == "t5xxl-abliterated", m
        assert m["text_encoder_clip"] == "flux-copy" and "text_encoder_clip_not_applied" not in m, m
        P.TEXT_ENCODER, P._TEXT_ENCODER_ACTIVE = dict(NONE), dict(NONE)
        m = P._gen_meta("txt2img", "p")
        assert not [k for k in m if k.startswith("text_encoder")], m
    finally:
        P.TEXT_ENCODER, P._TEXT_ENCODER_ACTIVE = old
    assert P._encoder_label(r"D:\m\flux-copy\text_encoder_2") == "flux-copy"
    assert P._encoder_label("/home/someone/enc/t5xxl-abl") == "t5xxl-abl"
    assert P._encoder_label("owner/repo/sub") == "owner/repo/sub"
    line = cz_imageio._a1111_parameters({"prompt": "p", "text_encoder_t5": "t5xxl-abliterated",
                                         "text_encoder_clip": "clip-gmp"})
    assert "Text encoder T5: t5xxl-abliterated" in line, line
    assert "Text encoder CLIP: clip-gmp" in line, line
    print("OK test_metadata_names_the_encoders_that_ran_and_never_their_paths")


def test_each_list_offers_the_folders_that_fit_its_slot():
    root = _mkdtemp("tes_")
    t5 = _folder(T5XXL, name="t5xxl-abliterated", root=root)
    clip = _folder(CLIPL, name="clip-gmp", root=root)
    flux = _folder(T5XXL, sub=T5, name="flux-krea-copy", root=root)   # copie diffusers
    _folder(CLIPL, sub=CLIP, name="flux-krea-copy", root=root)
    os.makedirs(os.path.join(root, "empty"))
    with _Dirs(te_dir=root):
        with _Base():
            got_t5, got_clip = P.list_text_encoders(T5), P.list_text_encoders(CLIP)
        with _Base({}):             # config du repo illisible hors ligne: pas de tri
            got_unsorted = P.list_text_encoders(T5)
    assert t5 in got_t5 and flux in got_t5 and clip not in got_t5, got_t5
    assert clip in got_clip and flux in got_clip and t5 not in got_clip, got_clip
    assert t5 in got_unsorted and clip in got_unsorted, got_unsorted
    assert not [f for f in got_t5 + got_clip + got_unsorted if f.endswith("empty")]
    # a cote du dossier de checkpoints EXTRA (bibliotheque sur un autre disque)
    lib = _mkdtemp("lib_")
    os.makedirs(os.path.join(lib, "checkpoints"))
    far = _folder(T5XXL, name="t5-far", root=os.path.join(lib, "text_encoders"))
    with _Dirs(extra=os.path.join(lib, "checkpoints")), _Base():
        assert far in P.list_text_encoders(T5)
    print("OK test_each_list_offers_the_folders_that_fit_its_slot")


class _FakeFlux:
    """Tient lieu de diffusers.FluxPipeline: retient ce que from_pretrained a recu."""
    calls = []

    @classmethod
    def from_pretrained(cls, repo, **kw):
        cls.calls.append((repo, kw))
        pipe = type("Pipe", (), {})()
        pipe.scheduler = type("S", (), {"config": {}})()
        pipe.vae = type("V", (), {"config": type("C", (), {})(),
                                   "enable_slicing": lambda self: None,
                                   "enable_tiling": lambda self: None})()
        pipe.to = lambda *a, **k: pipe
        pipe.enable_model_cpu_offload = lambda: None
        pipe.enable_sequential_cpu_offload = lambda: None
        return pipe


def test_ensure_base_passes_the_encoders_and_never_loses_the_render():
    import diffusers
    t5 = _folder(T5XXL, name="t5xxl-abliterated")
    clip = _folder(CLIPL, name="clip-broken")
    loaded, tried = object(), []

    def fake_load(src, component=T5, base=None):
        tried.append(component)
        if component == CLIP:
            raise OSError("safetensors header is corrupt")
        return loaded

    keys = ("TEXT_ENCODER", "_TEXT_ENCODER_ACTIVE", "_load_text_encoder", "_BASE_PIPE",
            "_DERIVED", "_LOADED_KEY", "_BASE_SCHED_CONFIG", "ZIMAGE_TRANSFORMER", "LORAS",
            "_APPLIED_LORAS")
    old = {k: getattr(P, k) for k in keys}
    old_flux = diffusers.FluxPipeline
    try:
        diffusers.FluxPipeline = _FakeFlux
        P._load_text_encoder = fake_load
        P.ZIMAGE_TRANSFORMER, P.LORAS = None, []
        P.free_vram()
        P.TEXT_ENCODER = {T5: t5, CLIP: clip}
        with _Base():
            P._ensure_base()
        _repo, kw = _FakeFlux.calls[-1]
        assert kw.get(T5) is loaded, "le T5 de remplacement doit etre passe a from_pretrained"
        assert CLIP not in kw, "un CLIP qui n'a pas pu se charger ne doit pas etre passe"
        assert sorted(tried) == [CLIP, T5], tried
        assert P._TEXT_ENCODER_ACTIVE == {T5: t5, CLIP: ""}, P._TEXT_ENCODER_ACTIVE
        m = P._gen_meta("txt2img", "p")
        assert m["text_encoder_t5"] == "t5xxl-abliterated", m
        assert m["text_encoder_clip_not_applied"] == "clip-broken", m
        # un T5 trop etroit: refuse a la config, jamais lu, et la generation continue
        tried.clear()
        P.free_vram()
        P.TEXT_ENCODER = {T5: _folder({**T5XXL, "d_model": 2048}, name="t5-xl"), CLIP: ""}
        with _Base():
            P._ensure_base()
        _repo, kw = _FakeFlux.calls[-1]
        assert tried == [] and T5 not in kw and CLIP not in kw, (tried, kw)
        assert not any(P._TEXT_ENCODER_ACTIVE.values()), P._TEXT_ENCODER_ACTIVE
        assert P._gen_meta("txt2img", "p")["text_encoder_t5_not_applied"] == "t5-xl"
    finally:
        diffusers.FluxPipeline = old_flux
        P.free_vram()
        for k, v in old.items():
            setattr(P, k, v)
    print("OK test_ensure_base_passes_the_encoders_and_never_loses_the_render")


def test_derived_pipes_share_the_encoders():
    """img2img et inpaint derivent du base par from_pipe (get_pipe): ils doivent reprendre
    SES encodeurs -- donc ceux de remplacement -- et non en recharger d'autres. Pipeline
    FLUX minuscule, poids aleatoires, CPU: rien n'est lu sur disque."""
    from diffusers import (AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline,
                           FluxTransformer2DModel)
    from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel
    torch.manual_seed(0)
    base = FluxPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=AutoencoderKL(sample_size=32, in_channels=3, out_channels=3,
                          block_out_channels=(4,), layers_per_block=1, latent_channels=1,
                          norm_num_groups=1, use_quant_conv=False, use_post_quant_conv=False),
        text_encoder=CLIPTextModel(CLIPTextConfig(
            hidden_size=32, intermediate_size=37, num_attention_heads=4,
            num_hidden_layers=2, vocab_size=1000, projection_dim=32)),
        tokenizer=None,
        text_encoder_2=T5EncoderModel(T5Config(
            vocab_size=1000, d_model=32, d_kv=8, d_ff=37, num_layers=2, num_heads=4)),
        tokenizer_2=None,
        transformer=FluxTransformer2DModel(
            patch_size=1, in_channels=4, num_layers=1, num_single_layers=1,
            attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
            pooled_projection_dim=32, axes_dims_rope=[4, 4, 8]),
    )
    keys = ("_BASE_PIPE", "_DERIVED", "_LOADED_KEY", "ZIMAGE_TRANSFORMER", "LORAS",
            "_APPLIED_LORAS", "_BASE_SCHED_CONFIG")
    old = {k: getattr(P, k) for k in keys}
    try:
        P.ZIMAGE_TRANSFORMER, P.LORAS, P._APPLIED_LORAS, P._BASE_SCHED_CONFIG = None, [], [], None
        P._BASE_PIPE, P._DERIVED = base, {"txt2img": base}
        P._LOADED_KEY = (P.BASE_REPO, None, P.OFFLOAD_MODE)
        for kind in ("img2img", "inpaint"):
            d = P.get_pipe(kind)
            assert d is not base and type(d).__name__ != "FluxPipeline", (kind, type(d))
            assert d.text_encoder is base.text_encoder, f"{kind}: CLIP not shared"
            assert d.text_encoder_2 is base.text_encoder_2, f"{kind}: T5 not shared"
    finally:
        for k, v in old.items():
            setattr(P, k, v)
    print("OK test_derived_pipes_share_the_encoders")


def test_the_ui_saves_only_an_encoder_that_fits():
    import cz_ui as U
    saved = []
    old = (dict(P.TEXT_ENCODER), U._save_prefs_keys)
    try:
        U._save_prefs_keys = saved.append          # jamais le vrai preferences.json
        P.TEXT_ENCODER = dict(NONE)
        with _Base(), _Dirs():
            msg = U._ui_set_text_encoder(CLIP, _folder(T5XXL))     # un T5 dans le slot CLIP
            assert "not applied" in msg and "CLIP" in msg, msg
            assert saved == [] and P.TEXT_ENCODER[CLIP] == "", saved
            good = _folder(T5XXL, name="t5xxl-abliterated")
            msg = U._ui_set_text_encoder(T5, good)
            assert saved == [{T5: good}] and P.TEXT_ENCODER[T5] == good, saved
            assert "t5xxl-abliterated" in msg and "reloads" in msg, msg
            U._ui_set_text_encoder(T5, "")        # retour au defaut: toujours permis
            assert saved[-1] == {T5: ""} and P.TEXT_ENCODER[T5] == "", saved
            # une valeur collee (repo HF) reste proposee dans le dropdown
            P.TEXT_ENCODER[CLIP] = "someone/clip-l-finetune"
            ch = U._te_choices(CLIP)
        assert ch[0] == ("Default (base repo's own CLIP)", ""), ch
        assert ("someone/clip-l-finetune", "someone/clip-l-finetune") in ch, ch
    finally:
        P.TEXT_ENCODER, U._save_prefs_keys = old
    print("OK test_the_ui_saves_only_an_encoder_that_fits")


def test_the_queue_keeps_the_encoders():
    import cz_ui as U
    calls = []
    old = (dict(P.TEXT_ENCODER), P.set_text_encoder)
    try:
        P.TEXT_ENCODER = {T5: r"D:\enc\t5xxl-abl", CLIP: ""}
        ms = U._q_model_state()
        assert ms[T5] == r"D:\enc\t5xxl-abl" and ms[CLIP] == "", ms
        P.set_text_encoder = lambda c, s: calls.append((c, s))
        U._q_restore_model_state(ms)
        assert sorted(calls) == [(CLIP, ""), (T5, r"D:\enc\t5xxl-abl")], calls
        # snapshot d'avant l'option: on ne touche pas aux encodeurs courants
        calls.clear()
        U._q_restore_model_state({k: v for k, v in ms.items() if k not in (T5, CLIP)})
        assert calls == [], calls
    finally:
        P.TEXT_ENCODER, P.set_text_encoder = old
    print("OK test_the_queue_keeps_the_encoders")


def test_the_config_sample_documents_the_keys():
    with open(os.path.join(ROOT, "config-sample.txt"), encoding="utf-8") as f:
        cfg = json.load(f)
    for k in ("text_encoder_2", "text_encoder", "text_encoders_dir"):
        assert cfg.get(k) == "" and cfg.get(f"_{k}_help"), k
    print("OK test_the_config_sample_documents_the_keys")


def test_default_picked_in_the_ui_survives_a_restart():
    """Choisir "Default" ecrit "" dans les preferences: au redemarrage, une valeur de
    config.txt ne doit pas revenir par-dessus. L'environnement gagne toujours."""
    old = (P._prefs, P.CONFIG)
    env = "KREA_TEXT_ENCODER_2_PROBE"
    try:
        P.CONFIG = {"text_encoder_2": r"D:\enc\from-config"}
        P._prefs = {}
        assert P._cfg_str("text_encoder_2", env) == r"D:\enc\from-config"
        P._prefs = {"text_encoder_2": ""}
        assert P._cfg_str("text_encoder_2", env) == ""
        P._prefs = {"text_encoder_2": r"D:\enc\ui"}
        assert P._cfg_str("text_encoder_2", env) == r"D:\enc\ui"
        os.environ[env] = r"D:\enc\env"
        P._prefs = {"text_encoder_2": ""}
        assert P._cfg_str("text_encoder_2", env) == r"D:\enc\env"
    finally:
        P._prefs, P.CONFIG = old
        os.environ.pop(env, None)
    print("OK test_default_picked_in_the_ui_survives_a_restart")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All text-encoder tests passed.")
