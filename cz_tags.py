"""crispz-studio - Tag autocomplete: sources cote serveur.

Telecharge UNE FOIS les CSV de tags (config tag_autocomplete.sources) dans tags/,
de facon atomique avec progression console. Tout .csv depose dans tags/ devient une
source cote client. Ce module n'est importe QUE si la feature est activee (contrat
zero-cout quand off); un echec reseau ne bloque jamais le boot (warning + continue).
"""

import os

from cz_core import HERE, _log, download_with_progress

TAGS_DIR = os.path.join(HERE, "tags")


def _source_filename(url):
    """Nom de fichier local d'une URL source (toujours .csv)."""
    from urllib.parse import urlparse
    name = os.path.basename(urlparse(str(url)).path.rstrip("/")) or "tags"
    if not name.lower().endswith(".csv"):
        name += ".csv"
    return name


def ensure_tag_sources(sources):
    """Telecharge chaque URL absente de tags/ (une fois). Echec = warning, on continue.
    Renvoie le nombre de fichiers telecharges."""
    os.makedirs(TAGS_DIR, exist_ok=True)
    done = 0
    for url in (sources or []):
        dst = os.path.join(TAGS_DIR, _source_filename(url))
        if os.path.isfile(dst):
            continue
        try:
            _log(f"downloading {os.path.basename(dst)} (first launch only)...", mod="tagac")
            download_with_progress(url, dst)
            done += 1
        except Exception as e:
            _log(f"download failed for {url} ({e}); continuing without", mod="tagac")
    return done


def list_tag_files():
    """Tous les .csv du dossier tags/ (sources du client)."""
    if not os.path.isdir(TAGS_DIR):
        return []
    return sorted(os.path.join(TAGS_DIR, f) for f in os.listdir(TAGS_DIR)
                  if f.lower().endswith(".csv"))
