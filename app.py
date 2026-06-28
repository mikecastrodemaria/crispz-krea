"""crispz-studio - Z-Image txt2img + upscaler/detailer (standalone, sans ComfyUI).

Point d'entree mince. Tout le code a ete decoupe en modules cz_* :
  cz_core (config/paths/logging/device) · cz_imageio (I/O image) · cz_prompt (styles/
  wildcards) · cz_ollama (describe/improve/compose) · cz_esrgan (Real-ESRGAN) ·
  cz_face (faceswap/restore/BLIP/rembg) · cz_pipeline (coeur Z-Image: generation,
  pipelines, LoRA/checkpoints, offload, guidance) · cz_assetbrowser / cz_assets ·
  cz_ui (build_ui + handlers) · cz_cli (argparse + serveur).

Ce fichier ne fait que (1) lancer la CLI/UI via cz_cli.cli_main et (2) re-exporter les
quelques symboles que tools/smoke_test.py lit via `import app`. L'etat mutable runtime
(LORAS / FACESWAP_RESTORE) est expose en proxy live par __getattr__.

Lancer:  python app.py            (UI)
         python app.py --help     (CLI)
"""

# Force l'implementation python de protobuf AVANT tout import (transformers/sentencepiece).
# Evite "TypeError: Descriptors cannot be created directly" au chargement du tokenizer T5
# (FLUX) avec protobuf 4.x+. Impact perf nul (protobuf sert juste au chargement tokenizer).
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys

# Modules conservant l'etat mutable runtime (lus en live par __getattr__).
import cz_pipeline
import cz_face
import cz_esrgan

# Re-exports pour le smoke et la retro-compat `import app` (noqa: symboles non utilises ici).
from cz_core import (  # noqa: F401
    CONFIG, COMPOSE_INSTRUCTION, IMPROVE_INSTRUCTION, DESCRIBE_INSTRUCTION,
    set_log_level,
)
from cz_prompt import STYLES, _apply_styles  # noqa: F401
from cz_imageio import _format_filename, save_image, _read_image_meta  # noqa: F401
from cz_pipeline import (  # noqa: F401
    _reframe_canvas, _gen_meta, set_loras, round_to_multiple,
    generate, txt2img_run, process_one, outpaint, inpaint_run, generate_omni,
)
from cz_face import set_faceswap_restore, _local_caption, _remove_bg  # noqa: F401
from cz_ui import (  # noqa: F401
    build_ui, run, _editor_img, _editor_to_image_mask, _gallery_load, _faceswap,
)
from cz_cli import cli_main, serve_main  # noqa: F401

main = cli_main


# Facade retro-compat: tout symbole deplace (ETAT MUTABLE inclus: LORAS, ESRGAN_DIR,
# BASE_REPO, FACESWAP_RESTORE, OFFLOAD_MODE, ...) reste accessible en live via app.NAME.
# Le smoke (app.LORAS / app.FACESWAP_RESTORE) et cli_interactive.py (app.ESRGAN_DIR /
# app.BASE_REPO / app.set_esrgan_dir ...) en dependent. Premier module qui matche gagne.
_PROXY_MODULES = (cz_pipeline, cz_esrgan, cz_face)


def __getattr__(name):
    for _m in _PROXY_MODULES:
        try:
            return getattr(_m, name)
        except AttributeError:
            continue
    raise AttributeError(f"module 'app' has no attribute {name!r}")


if __name__ == "__main__":
    sys.exit(cli_main())
