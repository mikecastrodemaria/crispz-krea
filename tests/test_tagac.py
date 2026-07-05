"""Unit tests for the tag-autocomplete server side (atomic download, sources, naming).

Run:  .venv/Scripts/python tests/test_tagac.py
"""
import http.server
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_tags  # noqa: E402
from cz_core import download_with_progress  # noqa: E402

PAYLOAD = b"1girl,0,5000000,\"1girls,sole_female\"\nsolo,0,4000000,\n" * 200


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ok.csv":
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD)
        else:
            self.send_error(404)

    def log_message(self, *a):
        pass


def _serve():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_download_atomic_ok():
    srv, base = _serve()
    tmp = tempfile.mkdtemp()
    dst = os.path.join(tmp, "ok.csv")
    out = download_with_progress(f"{base}/ok.csv", dst)
    assert out == dst and os.path.isfile(dst)
    assert open(dst, "rb").read() == PAYLOAD
    assert not os.path.exists(dst + ".tmp"), "tmp must be renamed away"
    srv.shutdown()


def test_download_failure_cleans_tmp():
    srv, base = _serve()
    tmp = tempfile.mkdtemp()
    dst = os.path.join(tmp, "missing.csv")
    try:
        download_with_progress(f"{base}/missing.csv", dst)
        raise AssertionError("expected failure")
    except AssertionError:
        raise
    except Exception:
        pass
    assert not os.path.exists(dst), "no truncated file served"
    assert not os.path.exists(dst + ".tmp"), "tmp cleaned on failure"
    srv.shutdown()


def test_source_filename():
    f = cz_tags._source_filename
    assert f("https://x.y/tags/danbooru.csv") == "danbooru.csv"
    assert f("https://x.y/tags/danbooru.csv?raw=1") == "danbooru.csv"
    assert f("https://x.y/wordlist") == "wordlist.csv"
    assert f("https://x.y/") == "tags.csv"


def test_ensure_sources_resilient(monkey_dir=None):
    srv, base = _serve()
    old = cz_tags.TAGS_DIR
    cz_tags.TAGS_DIR = tempfile.mkdtemp()
    try:
        # 1 URL valide + 1 cassee -> 1 fichier, pas d'exception
        n = cz_tags.ensure_tag_sources([f"{base}/ok.csv", f"{base}/broken.csv"])
        assert n == 1
        assert [os.path.basename(p) for p in cz_tags.list_tag_files()] == ["ok.csv"]
        # deja present -> pas de re-telechargement
        assert cz_tags.ensure_tag_sources([f"{base}/ok.csv"]) == 0
    finally:
        cz_tags.TAGS_DIR = old
        srv.shutdown()


def test_list_filters_csv():
    old = cz_tags.TAGS_DIR
    cz_tags.TAGS_DIR = tempfile.mkdtemp()
    try:
        for name in ("a.csv", "B.CSV", "notes.txt"):
            open(os.path.join(cz_tags.TAGS_DIR, name), "w").close()
        names = [os.path.basename(p) for p in cz_tags.list_tag_files()]
        assert names == ["B.CSV", "a.csv"]
    finally:
        cz_tags.TAGS_DIR = old


if __name__ == "__main__":
    for fn in (test_download_atomic_ok, test_download_failure_cleans_tmp,
               test_source_filename, test_ensure_sources_resilient, test_list_filters_csv):
        fn()
        print(f"OK {fn.__name__}")
    print("All tagac tests passed.")
