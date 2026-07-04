# Changelog

All notable changes to crispz-studio. One versioned entry per feature.
The app version lives in `cz_core.py` (`APP_VERSION`) and is shown in the browser tab title.

## 1.1.0 — 2026-07-04 — Job queue

Queue up generations with different settings and run them unattended (e.g. overnight).

- **`+ Queue`** snapshots ALL current settings: the full Generate parameter set **plus the
  global model state** (checkpoint/transformer, active LoRAs + weights, sampler, schedule),
  so each job is self-contained and reproducible regardless of what is loaded later.
  The button label shows the pending count (`+ Queue (3)`).
- **Job queue panel** (accordion under the prompt area): readable labels
  (`txt2img · model · 1024x768 · 8 steps · seed 42 · x2 · "prompt…"`), select a job and
  **Up / Down / Remove / Clear**.
- **`Run queue`** executes jobs in order in the normal progress window; the session
  history and saved outputs accumulate as usual. Before each job the model state is
  restored through the existing setters, so **VRAM is purged automatically only when the
  model actually changes** between jobs (zero cost otherwise).
- **Stop pauses the queue**: the current job is interrupted (existing Stop behavior) and
  the remaining jobs stay queued — press `Run queue` again to resume. A failing job is
  logged (`[crispz][queue] …`) and the queue continues with the next one.
- Config block (`config.txt`): `"job_queue": {"enabled": true}` — set `false` to remove
  the panel entirely (no components, no handlers, zero cost).
- Files: `cz_ui.py` (panel + handlers + pure helpers), `cz_core.py` (`APP_VERSION`,
  module-prefixed logs), `config-sample.txt`, `tests/test_queue.py`.
- Limits (v1): the queue lives in memory (cleared on page reload); jobs are not editable
  in place (remove + re-queue); execution is sequential.

## 1.0.0 — 2026-07-04 — Baseline

Everything up to and including: unified Inpaint/Outpaint editor (brush / expand sides /
reframe, ~1 MP bound, harmonize), auto-upscale after generate, local BLIP captioner +
auto-describe, unified Z-Image checkpoint dropdown (+ extra folder, Performance
auto-sync), multi-LoRA, face swap + GFPGAN, remove background, Asset Browser (instant
open, day filter, placeholders), Ollama integration with offline fallbacks, CLI and
server mode.
