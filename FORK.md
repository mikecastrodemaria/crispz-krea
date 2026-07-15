# crispz-krea — fork de crispz-studio

Ce dépôt est un **fork de [crispz-studio](https://github.com/mikecastrodemaria/crispz-studio)**
dont le moteur de génération est remplacé par **FLUX.1 Krea [dev]**.

- **Cible** : txt2img haut de gamme avec `FluxPipeline` (diffusers), ~12B, rendu esthétique « Krea ».
- **Modèle HF** : `black-forest-labs/FLUX.1-Krea-dev` (**repo gated** → login HF requis au 1er download).

## Ce qui change vs l'upstream

Le SEUL fichier modèle à réécrire est **`cz_pipeline.py`** :

- Loader → `FluxPipeline.from_pretrained(..., torch_dtype=bfloat16)` ; img2img → `FluxImg2ImgPipeline` ; inpaint/outpaint → `FluxInpaintPipeline`.
- Scheduler natif `FlowMatchEulerDiscreteScheduler` (remplace le schedule de sigmas custom Z-Image).
- Guidance : Krea **dev** est guidance-distillé → `guidance_scale` réel (~3.5–5), pas 0.
- LoRA → `pipe.load_lora_weights(...)` / `set_adapters` (format Flux).
- `generate_omni` (édition multi-référence) → non supporté : raise + onglet masqué.

Tout le reste (`cz_ui`, `cz_face`, `cz_esrgan`, `cz_ollama`, styles, CLI) est **conservé tel quel**.

## Workflow upstream (correctifs d'infra commune)

```
git fetch upstream
git merge upstream/main
```

`upstream` = crispz-studio (Z-Image). `origin` = ce fork.

**Conflits attendus : 4 fichiers** (pas seulement `cz_pipeline.py`). Règle de résolution :

| Fichier | Résolution | Delta fork à préserver |
|---|---|---|
| `cz_pipeline.py` | **ours** (couche modèle FLUX) + porter à la main les améliorations génériques d'upstream | tout le loader Flux/GGUF |
| `cz_ui.py` | **theirs** (upstream) + réappliquer le delta fork | `ZIMAGE_BASE_REPOS`, `ZIMAGE_BASE_PERFORMANCE`, libellés « FLUX … » |
| `cz_core.py` | **theirs** + réappliquer | `DEFAULT_BASE_REPO`, `MODEL_PROFILES`/`DEFAULT_MODEL_PROFILE`, `.gguf` dans `_is_single_file` |
| `config-sample.txt` | **theirs** + réappliquer | presets/profils Krea, `default_*`, `flux_true_cfg` |

### Ce qui se porte depuis upstream vers `cz_pipeline.py` (générique)

`_load_monitor`/`_fmt_load`/`_load_pct` (progression de chargement), `_apply_loras` +
`_APPLIED_LORAS` (hot-swap LoRA PEFT — `set_loras` ne doit **pas** appeler `free_vram`),
`_lora_weight_range` + `LORA_WEIGHT_MIN/MAX`, `_swap_transformer`/`_load_transformer`
(recharger le transformer seul), `_LAST_SEED`/`_NO_SEED_INCREMENT`/`_SAVE_PRE_UPSCALE`
+ leurs setters (requis par `cz_ui`).

### Ce qui NE se porte PAS (spécifique Z-Image)

`round_to_multiple(x, m=32)` → **Flux reste en `m=16`** (VAE 8 × patch 2). Idem le snap /32
de `_refine_whole`, et tout import `ZImagePipeline`/`ZImageTransformer2DModel`.

### Après merge, vérifier

Contrat d'API (`cz_ui`/`cz_cli` importent tout ce que `cz_pipeline` doit exposer),
`cz_ui.build_ui()` headless, `round_to_multiple(100) == 96`, defaults Krea intacts,
`tests/` (surtout `test_lora_hotswap`, `test_load`, `test_model_swap`).

## État

- [x] Réécrire `cz_pipeline.py` pour FLUX.1 Krea (`FluxPipeline` / `FluxImg2ImgPipeline` /
      `FluxInpaintPipeline` / `FluxTransformer2DModel`, guidance distillée 4.5, CFG réel
      conditionnel via `true_cfg_scale`). API publique du module **inchangée** (cz_ui/cz_cli
      intacts). `generate_omni` lève une erreur claire (pas d'Omni en txt2img).
- [x] Rebrander les défauts Krea : `config-sample.txt` (guidance 4.5, 28 steps, presets,
      `model_profiles` flux/krea/schnell) + `cz_core.py` (`DEFAULT_BASE_REPO`, profils).
- [x] Vérifié : `import app` / `import cz_ui` OK, contrat d'API complet préservé, defaults
      corrects (BASE_REPO=Krea, GUIDANCE=4.5, profil=(28,4.5)).
- [ ] `requirements.txt` : FluxPipeline est déjà dans diffusers (source). Rien à ajouter a
      priori — à confirmer au 1er run réel.
- [ ] Étape token HF à l'install (repo **gated** `black-forest-labs/FLUX.1-Krea-dev`).
- [ ] Masquer l'onglet édition/Omni côté UI (optionnel : il lève déjà une erreur claire).
- [ ] Test génération réel sur GPU (download du modèle gated).
- [ ] Mettre à jour README + identité (titres, captures).
- [ ] Launcher Pinokio `crispz-krea.pinokio.git` (clone ce repo, env `HF_TOKEN`).
