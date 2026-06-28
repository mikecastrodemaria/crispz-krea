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
git merge upstream/main      # seuls les conflits attendus sont dans cz_pipeline.py
```

`upstream` = crispz-studio (Z-Image). `origin` = ce fork.

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
