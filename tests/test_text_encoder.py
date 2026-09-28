"""Replacement text encoders (Models > Checkpoints > Text encoder T5 / CLIP).

FLUX.1 Krea has TWO encoders, each with its diffusers component and its own choice:
  text_encoder_2 = T5-XXL (T5EncoderModel) -> prompt_embeds, the sequence the
                   transformer reads (4096 wide, 24 layers);
  text_encoder   = CLIP-L (CLIPTextModel)  -> pooled_prompt_embeds (768, 12 layers).
Another encoder only plugs in when it has the same family, the same width and the same
number of layers as the SAME component of the base repo, and the refusal must say so BEFORE
reading 9 GB of T5.

These tests also lock down what would make the option silently dangerous:
  - a change of encoder empties the embeddings cache, and BOTH encoders are in
    the cache KEY (it only held id(pipe.text_encoder), the CLIP, whereas
    it is the T5 that produces prompt_embeds);
  - _ensure_base passes the encoder to from_pretrained and, at the slightest problem, falls
    back on the repo's without losing the generation;
  - the derived pipes (img2img, inpaint: from_pipe) share the base's encoders;
  - the metadata names the encoders that REALLY ran, by their folder
    name and never by their path (which would end up in the shared PNGs);
  - the UI only remembers a valid encoder; the queue keeps the job's.

No real model, no network, no GPU: dummy configs, dummy or tiny pipelines
(a few KB of random weights).

Run:  .venv/Scripts/python tests/test_text_encoder.py

"""
import json
import os
import sys
import tempfile

# Never any network, not even by accident: huggingface_hub reads this at import time.
os.environ["HF_HUB_OFFLINE"] = "1"
# Never any GPU either: get_pipe would move the tiny pipeline onto cuda. '-1' and
# not '': under Windows, an empty variable does not hide the GPU.
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
BASE = {T5: T5XXL, CLIP: CLIPL}          # what black-forest-labs/FLUX.1-Krea-dev says
NONE = dict.fromkeys(P.TEXT_ENCODER_COMPONENTS, "")

_TMP = tempfile.TemporaryDirectory(prefix="crispz_te_")   # everything is erased on the way out


def _mkdtemp(prefix):
    return tempfile.mkdtemp(prefix=prefix, dir=_TMP.name)


def _folder(cfg, sub=None, name="enc", root=None):
    """A dummy encoder folder: one config.json, at the root or in `sub`."""
    d = os.path.join(root or _mkdtemp("te_"), name)
    p = os.path.join(d, sub) if sub else d
    os.makedirs(p, exist_ok=True)
    with open(os.path.join(p, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return d


class _Base:
    """Replaces the reading of the base repo's encoder configs (no network, no
    HF) and remembers which components were asked for."""

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
    """The listings only see `te_dir` and `extra`, never the machine's real folders."""

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
        # a copy of a diffusers repo: the T5 is in text_encoder_2/, the CLIP in text_encoder/
        assert P._text_encoder_problem(_folder(T5XXL, T5), T5) is None
        assert P._text_encoder_problem(_folder(CLIPL, CLIP), CLIP) is None
        assert P._text_encoder_problem(_folder(CLIPL), CLIP) is None
        # a complete T5ForConditionalGeneration: T5EncoderModel only reads its encoder
        full_t5 = {**T5XXL, "architectures": ["T5ForConditionalGeneration"],
                   "num_decoder_layers": 24}
        assert P._text_encoder_problem(_folder(full_t5), T5) is None
        # a complete CLIPModel (text + vision), the usual shape of a fine-tuned CLIP-L
        full_clip = {"model_type": "clip", "architectures": ["CLIPModel"],
                     "text_config": {"hidden_size": 768, "num_hidden_layers": 12},
                     "vision_config": {"hidden_size": 1024, "num_hidden_layers": 24}}
        assert P._text_encoder_problem(_folder(full_clip), CLIP) is None
    print("OK test_same_architecture_is_accepted")


def test_each_slot_is_checked_against_its_own_base_component():
    """The T5 is compared with the repo's text_encoder_2, the CLIP with the text_encoder: the same
    folder suits one and not the other."""
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
        # a path from another machine is never taken for an HF repo
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
    # local_only: a repo absent from the cache returns None, with no network
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
        assert P._BASE_PIPE is None, "the pipeline must be released"
        assert not P._EMBED_CACHE, "the old encodings would still be served"
        assert not any(P._TEXT_ENCODER_ACTIVE.values()), "free_vram kept the old encoder"
        # the same value: nothing moves, no pointless reload
        sentinel = P._BASE_PIPE = object()
        P.set_text_encoder(T5, r"D:\enc\t5xxl-abl")
        assert P._BASE_PIPE is sentinel
        # the other component changes apart, and frees as well
        P.set_text_encoder(CLIP, r"D:\enc\clip-gmp")
        assert P._BASE_PIPE is None and P.TEXT_ENCODER[T5] == r"D:\enc\t5xxl-abl"
        try:
            P.set_text_encoder("text_encoder_3", "x")
            raise AssertionError("an unknown component must be refused")
        except ValueError:
            pass
    finally:
        P.TEXT_ENCODER, P._BASE_PIPE = old
        P._TEXT_ENCODER_ACTIVE = dict(NONE)
        P._EMBED_CACHE.clear()
    print("OK test_changing_an_encoder_frees_the_pipe_and_the_cache")


class FakePipe:
    """Two encoders, like FluxPipeline; it counts the encodings."""

    def __init__(self):
        self.text_encoder = object()        # CLIP
        self.text_encoder_2 = object()      # T5
        self._execution_device = "cpu"
        self.n = 0

    def encode_prompt(self, prompt=None, device=None, **kw):
        self.n += 1
        return (torch.zeros(1, 2, 4), torch.zeros(1, 4), torch.zeros(2, 3))


def test_the_embed_key_carries_both_encoders():
    """The same prompt, the same pipe: another T5 OR another CLIP = another encoding."""
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
        assert pipe.n == 2, "an encoding from the old T5 was served again"
        P._TEXT_ENCODER_ACTIVE = {T5: r"D:\enc\t5xxl-abl", CLIP: r"D:\enc\clip-gmp"}
        P._cached_prompt_embeds(pipe, "p", {})
        assert pipe.n == 3, "an encoding from the old CLIP was served again"
        # the pipe's T5 replaced by another object: the old key only saw the CLIP
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
        assert "someone" not in json.dumps(m), "a local path in the metadata"
        # the T5 discarded at load time, the CLIP applied: each one its field
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
    flux = _folder(T5XXL, sub=T5, name="flux-krea-copy", root=root)   # a diffusers copy
    _folder(CLIPL, sub=CLIP, name="flux-krea-copy", root=root)
    os.makedirs(os.path.join(root, "empty"))
    with _Dirs(te_dir=root):
        with _Base():
            got_t5, got_clip = P.list_text_encoders(T5), P.list_text_encoders(CLIP)
        with _Base({}):             # the repo's config unreadable offline: no sorting
            got_unsorted = P.list_text_encoders(T5)
    assert t5 in got_t5 and flux in got_t5 and clip not in got_t5, got_t5
    assert clip in got_clip and flux in got_clip and t5 not in got_clip, got_clip
    assert t5 in got_unsorted and clip in got_unsorted, got_unsorted
    assert not [f for f in got_t5 + got_clip + got_unsorted if f.endswith("empty")]
    # next to the EXTRA checkpoints folder (a library on another disk)
    lib = _mkdtemp("lib_")
    os.makedirs(os.path.join(lib, "checkpoints"))
    far = _folder(T5XXL, name="t5-far", root=os.path.join(lib, "text_encoders"))
    with _Dirs(extra=os.path.join(lib, "checkpoints")), _Base():
        assert far in P.list_text_encoders(T5)
    print("OK test_each_list_offers_the_folders_that_fit_its_slot")


class _FakeFlux:
    """Stands in for diffusers.FluxPipeline: it remembers what from_pretrained received."""
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
        assert kw.get(T5) is loaded, "the replacement T5 must be passed to from_pretrained"
        assert CLIP not in kw, "a CLIP that failed to load must not be passed"
        assert sorted(tried) == [CLIP, T5], tried
        assert P._TEXT_ENCODER_ACTIVE == {T5: t5, CLIP: ""}, P._TEXT_ENCODER_ACTIVE
        m = P._gen_meta("txt2img", "p")
        assert m["text_encoder_t5"] == "t5xxl-abliterated", m
        assert m["text_encoder_clip_not_applied"] == "clip-broken", m
        # a T5 that is too narrow: refused at the config, never read, and the generation carries on
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
    """img2img and inpaint derive from the base through from_pipe (get_pipe): they must take
    ITS encoders -- so the replacement ones -- and not reload others. A tiny
    FLUX pipeline, random weights, the CPU: nothing is read from the disk."""
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
        U._save_prefs_keys = saved.append          # never the real preferences.json
        P.TEXT_ENCODER = dict(NONE)
        with _Base(), _Dirs():
            msg = U._ui_set_text_encoder(CLIP, _folder(T5XXL))     # a T5 in the CLIP slot
            assert "not applied" in msg and "CLIP" in msg, msg
            assert saved == [] and P.TEXT_ENCODER[CLIP] == "", saved
            good = _folder(T5XXL, name="t5xxl-abliterated")
            msg = U._ui_set_text_encoder(T5, good)
            assert saved == [{T5: good}] and P.TEXT_ENCODER[T5] == good, saved
            assert "t5xxl-abliterated" in msg and "reloads" in msg, msg
            U._ui_set_text_encoder(T5, "")        # back to the default: always allowed
            assert saved[-1] == {T5: ""} and P.TEXT_ENCODER[T5] == "", saved
            # a pasted value (an HF repo) stays offered in the dropdown
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
        # a snapshot from before the option: we do not touch the current encoders
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
    """Choosing "Default" writes "" into the preferences: on a restart, a value from
    config.txt must not come back over it. The environment always wins."""
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


def test_compatible_encoders_in_the_hf_cache_are_listed_per_component():
    """An encoder downloaded from HF lives in the HF cache: ITS component's list must
    show it. Not a diffusers pipeline, not a config with no weights, not another size,
    not a config with no size (a VAE, an upscaler), and a complete CLIP counts for the CLIP."""
    import json as _json
    import os as _os
    import tempfile as _tempfile
    t5 = {"model_type": "t5", "d_model": 64, "num_layers": 2}
    clip_text = {"model_type": "clip_text_model", "hidden_size": 32, "num_hidden_layers": 3}
    root = _tempfile.mkdtemp(prefix="hfcache_")

    def snap(repo, sub=None, cfg=t5, weights=True, pipeline=False):
        d = _os.path.join(root, "models--" + repo.replace("/", "--"), "snapshots", "r1")
        p = _os.path.join(d, sub) if sub else d
        _os.makedirs(p, exist_ok=True)
        with open(_os.path.join(p, "config.json"), "w", encoding="utf-8") as f:
            _json.dump(cfg, f)
        if weights:
            open(_os.path.join(p, "model.safetensors"), "wb").close()
        if pipeline:
            with open(_os.path.join(d, "model_index.json"), "w", encoding="utf-8") as f:
                f.write("{}")

    snap("a/t5-fits")
    snap("b/t5-in-sub", sub="enc")
    snap("c/t5-wider", cfg={"model_type": "t5", "d_model": 128, "num_layers": 2})
    snap("d/pipeline", sub="text_encoder_2", pipeline=True)
    snap("e/config-only", weights=False)
    snap("f/no-shape", cfg={"_class_name": "AutoencoderKL"})
    snap("g/full-clip", cfg={"model_type": "clip", "text_config": {
        "model_type": "clip_text_model", "hidden_size": 32, "num_hidden_layers": 3}})
    refs = {"text_encoder_2": t5, "text_encoder": clip_text}
    old = (P._hf_cache_dir, P._base_text_encoder_config)
    try:
        P._hf_cache_dir = lambda: root
        P._base_text_encoder_config = lambda base=None, component="text_encoder_2", **k: refs[component]
        got_t5 = [v for _l, v in P.list_cached_text_encoders("text_encoder_2")]
        got_clip = [v for _l, v in P.list_cached_text_encoders("text_encoder")]
        import cz_ui as U
        choices = [v for _l, v in U._te_choices("text_encoder_2")]
    finally:
        P._hf_cache_dir, P._base_text_encoder_config = old
    assert got_t5 == ["a/t5-fits", "b/t5-in-sub/enc"], got_t5
    assert got_clip == ["g/full-clip"], got_clip
    assert all(v in choices for v in got_t5), choices
    print("OK test_compatible_encoders_in_the_hf_cache_are_listed_per_component")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All text-encoder tests passed.")
