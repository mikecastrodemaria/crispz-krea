"""Unit tests for the LoRA hot-swap logic (_apply_loras) — no model is loaded.

Regression guard for: activating a LoRA used to call free_vram() and reload the whole
Z-Image pipeline (transformer + VAE + Qwen3 encoder, ~50s+). LoRAs must now be swapped
on the cached pipe instead.

Run:  .venv/Scripts/python tests/test_lora_hotswap.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as P  # noqa: E402


class FakePipe:
    """Records the LoRA calls diffusers would make, modelling the life cycle of the
    PEFT adapters: unload_lora_weights does NOT empty the registry (an observed
    behaviour that triggers 'Already found a peft_config'); only delete_adapters does."""

    def __init__(self, fail=False, unload_clears=False):
        self.calls = []
        self.fail = fail
        self.unload_clears = unload_clears
        self.adapters = {}          # name -> weight_name (a simulated PEFT registry)

    def unload_lora_weights(self):
        self.calls.append(("unload",))
        if self.unload_clears:
            self.adapters.clear()

    def get_list_adapters(self):
        return {"transformer": list(self.adapters)} if self.adapters else {}

    def delete_adapters(self, names):
        names = [names] if isinstance(names, str) else list(names)
        self.calls.append(("delete", tuple(names)))
        for n in names:
            self.adapters.pop(n, None)

    def load_lora_weights(self, folder, weight_name=None, adapter_name=None, **kw):
        if self.fail:
            raise RuntimeError("PEFT backend is required")
        self.calls.append(("load", weight_name, adapter_name))
        self.adapters[adapter_name] = weight_name

    def set_adapters(self, names, weights):
        self.calls.append(("set_adapters", tuple(names), tuple(weights)))


def _lora_file(d, name):
    p = os.path.join(d, name)
    with open(p, "wb") as f:
        f.write(b"x")
    return p


def _reset(applied, loras):
    P._APPLIED_LORAS = list(applied)
    P.LORAS = list(loras)


def test_no_change_is_a_noop():
    d = tempfile.mkdtemp()
    a = _lora_file(d, "a.safetensors")
    _reset([(a, 1.0)], [(a, 1.0)])
    pipe = FakePipe()
    assert P._apply_loras(pipe) is True
    assert pipe.calls == [], "rien ne doit bouger si la combinaison est identique"


def test_weight_only_change_uses_set_adapters():
    d = tempfile.mkdtemp()
    a = _lora_file(d, "a.safetensors")
    _reset([(a, 1.0)], [(a, 0.4)])
    pipe = FakePipe()
    assert P._apply_loras(pipe) is True
    # only a re-weighting: no unload, no reloading of the file
    assert pipe.calls == [("set_adapters", ("cz_lora_0",), (0.4,))]
    assert P._APPLIED_LORAS == [(a, 0.4)]


def test_different_loras_unload_then_reload():
    d = tempfile.mkdtemp()
    a, b = _lora_file(d, "a.safetensors"), _lora_file(d, "b.safetensors")
    _reset([(a, 1.0)], [(b, 0.8)])
    pipe = FakePipe()
    assert P._apply_loras(pipe) is True
    assert pipe.calls[0] == ("unload",)
    assert ("load", "b.safetensors", "cz_lora_0") in pipe.calls
    assert ("set_adapters", ("cz_lora_0",), (0.8,)) in pipe.calls
    assert P._APPLIED_LORAS == [(b, 0.8)]


def test_removing_all_loras_unloads():
    d = tempfile.mkdtemp()
    a = _lora_file(d, "a.safetensors")
    _reset([(a, 1.0)], [])
    pipe = FakePipe()
    assert P._apply_loras(pipe) is True
    assert pipe.calls == [("unload",)]          # nothing left to apply
    assert P._APPLIED_LORAS == []


def test_missing_file_is_ignored():
    _reset([], [("D:/nope/ghost.safetensors", 1.0)])
    pipe = FakePipe()
    assert P._apply_loras(pipe) is True
    assert not any(c[0] == "load" for c in pipe.calls)


def test_swap_leaves_only_the_new_adapter_registered():
    """The 'Already found a peft_config' regression: after an A -> B swap, the PEFT registry
    must hold ONLY B. As unload_lora_weights does not empty it (unload_clears=False),
    _clear_loras must delete the old adapter explicitly before reloading."""
    d = tempfile.mkdtemp()
    a, b = _lora_file(d, "a.safetensors"), _lora_file(d, "b.safetensors")
    pipe = FakePipe(unload_clears=False)      # like diffusers 0.39 (a residual peft_config)
    # state 1: A applied
    _reset([], [(a, 1.0)])
    assert P._apply_loras(pipe, force=True) is True
    assert set(pipe.adapters) == {"cz_lora_0"} and pipe.adapters["cz_lora_0"] == "a.safetensors"
    # state 2: a swap to B
    _reset([(a, 1.0)], [(b, 0.8)])
    assert P._apply_loras(pipe) is True
    assert ("delete", ("cz_lora_0",)) in pipe.calls        # the old one really was deleted
    assert set(pipe.adapters) == {"cz_lora_0"} and pipe.adapters["cz_lora_0"] == "b.safetensors"
    # no accumulation: a single adapter, pointing at B
    assert len(pipe.adapters) == 1


def test_removing_all_clears_the_registry():
    d = tempfile.mkdtemp()
    a = _lora_file(d, "a.safetensors")
    pipe = FakePipe(unload_clears=False)
    _reset([], [(a, 1.0)]); P._apply_loras(pipe, force=True)
    _reset([(a, 1.0)], [])                     # we remove every LoRA
    assert P._apply_loras(pipe) is True
    assert pipe.adapters == {}                 # an empty registry, nothing left lying around


def test_failure_returns_false_for_full_reload_fallback():
    d = tempfile.mkdtemp()
    a = _lora_file(d, "a.safetensors")
    _reset([], [(a, 1.0)])
    pipe = FakePipe(fail=True)
    assert P._apply_loras(pipe) is False        # -> the caller will do free_vram + a reload
    assert P._APPLIED_LORAS == []


def test_set_loras_does_not_free_the_pipe():
    """The heart of the fix: set_loras must NO LONGER invalidate the loaded pipeline."""
    d = tempfile.mkdtemp()
    a = _lora_file(d, "a.safetensors")
    P.LORAS = []
    sentinel = object()
    P._BASE_PIPE = sentinel
    P._LOADED_KEY = ("repo", None, "none")
    P.set_loras([(a, 0.7)])
    assert P.LORAS == [(a, 0.7)]
    assert P._BASE_PIPE is sentinel, "set_loras ne doit pas liberer le pipe (pas de reload)"
    assert P._LOADED_KEY == ("repo", None, "none")
    P._BASE_PIPE = None


def test_base_cache_key_excludes_loras():
    """The cache key must no longer depend on the LoRAs (otherwise a reload on every change)."""
    import inspect
    src = inspect.getsource(P._ensure_base)
    assert "tuple(LORAS)" not in src, "les LoRA ne doivent pas faire partie de la cle de cache"


if __name__ == "__main__":
    for fn in (test_no_change_is_a_noop, test_weight_only_change_uses_set_adapters,
               test_different_loras_unload_then_reload, test_removing_all_loras_unloads,
               test_missing_file_is_ignored, test_swap_leaves_only_the_new_adapter_registered,
               test_removing_all_clears_the_registry,
               test_failure_returns_false_for_full_reload_fallback,
               test_set_loras_does_not_free_the_pipe, test_base_cache_key_excludes_loras):
        fn()
        print(f"OK {fn.__name__}")
    print("All lora hot-swap tests passed.")
