#!/usr/bin/env python3
"""glm_worker.py (v2) - tag-driven GLM-OCR worker for paperless-ngx.

For every document carrying the trigger tag it:
  1. downloads the original file from paperless-ngx,
  2. runs the GLM-OCR SDK (layout detection + region OCR) via its CLI,
  3. separately reads the header/footer strips the layout model skipped
     (logos, form numbers, copyright lines, page numbers),
  4. writes the combined plain text back into the document's content,
  5. swaps the trigger tag for NEXT_TAG (default glm-ocr-done) on success, or for FAIL_TAG
     (default glm-ocr-failed) after MAX_ATTEMPTS failed tries.
"""
import base64
import html
import json
import logging
import os
import hashlib
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from logging.handlers import RotatingFileHandler

import pymupdf
import requests

# ---------------------------------------------------------------- settings
URL = os.environ.get("PAPERLESS_URL", "").rstrip("/")
TOKEN = os.environ.get("PAPERLESS_TOKEN", "")
TRIGGER = os.environ.get("TRIGGER_TAG", "glm-ocr")
NEXT = os.environ.get("NEXT_TAG", "glm-ocr-done")  # added on success
FAIL = os.environ.get("FAIL_TAG", "glm-ocr-failed")  # added after MAX_ATTEMPTS failures
POLL = int(os.environ.get("POLL_SECONDS", "60"))
VERBOSE = os.environ.get("VERBOSE", "0") == "1"  # log every poll, even when idle
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "3"))
MIN_CHARS_PER_PAGE = int(os.environ.get("MIN_CHARS_PER_PAGE", "40"))
STATE = pathlib.Path(os.environ.get("STATE_FILE", "worker_state.json"))
BACKUPS = pathlib.Path(os.environ.get("BACKUP_DIR", "content_backups"))

OCR_HOST = os.environ.get("OCR_HOST", "localhost")
OCR_PORT = os.environ.get("OCR_PORT", "11434")
OCR_MODEL = os.environ.get("OCR_MODEL", "glm-ocr")
CHUNK_PAGES = int(os.environ.get("CHUNK_PAGES", "25"))          # pages per layout run; 0 = whole file at once
CHUNK_TIMEOUT = int(os.environ.get("CHUNK_TIMEOUT_SECONDS", "1800"))  # time limit per chunk
CACHE = pathlib.Path(os.environ.get("CACHE_DIR", str(pathlib.Path(os.environ.get("STATE_FILE", "worker_state.json")).resolve().parent / "chunk_cache")))
HEARTBEAT_SECONDS = int(os.environ.get("HEARTBEAT_SECONDS", "120"))  # progress line while a chunk runs
WARM_KEEP_ALIVE = os.environ.get("WARM_KEEP_ALIVE", "2m")
OCR_URL = f"http://{OCR_HOST}:{OCR_PORT}/v1/chat/completions"
LAYOUT_DEVICE = os.environ.get("LAYOUT_DEVICE", "cpu")
GLMOCR_BIN = str(pathlib.Path(sys.executable).parent / "glmocr")

TOP_MIN, BOTTOM_MAX = 25, 975  # layout coords are 0..1000; ignore thinner gaps
MAX_STRIP_CHARS = 8000         # hard backstop; shorter strips are kept unless they look like a loop
INK_MIN_PIXELS = 300           # fewer dark pixels than this = blank strip (200 dpi)

# Text that is interface noise from fillable PDFs, not document content.
NOISE = [r"Click\s*&\s*Tab\s*To\s*Begin", r"Reset\s*Form"]

HDR = {"Authorization": f"Token {TOKEN}", "Accept": "application/json"}
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("glm-worker")
if os.environ.get("LOG_FILE"):
    _fh = RotatingFileHandler(os.environ["LOG_FILE"], maxBytes=5_000_000, backupCount=3)
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(_fh)


# ------------------------------------------------------------- text helpers
def flatten_tables(text: str) -> str:
    """Turn the SDK's HTML tables into plain rows, cells separated by ' | '."""
    text = re.sub(r"</tr\s*>", "\n", text, flags=re.I)
    text = re.sub(r"</t[dh]\s*>", " | ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"(?m)(?:[ \t]*\|)+[ \t]*$", "", text)  # trailing empty cells
    text = re.sub(r"(?m)^[ \t|]*$", "", text)              # rows with no content
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_region(content: str) -> str:
    text = flatten_tables(content or "")
    return re.sub(r"(?m)^#{1,6}\s*", "", text).strip()


def drop_noise(text: str) -> str:
    for pattern in NOISE:
        text = re.sub(pattern, " ", text, flags=re.I)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


# ----------------------------------------------------------- strip reading
def loop_flag(text: str) -> str:
    """Return a reason if text looks like a model repetition loop, else ''."""
    if re.search(r"(.)\1{19,}", text):
        return "character run"
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= 8 and len(set(lines)) / len(lines) < 0.4:
        return "repeated lines"
    words = text.split()
    if len(words) >= 60:
        grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
        if len(set(grams)) / len(grams) < 0.3:
            return "repeated phrases"
    return ""


def read_strip(page, y0: float, y1: float) -> str:
    """OCR one horizontal band of a page (y0..y1 on a 0..1000 scale)."""
    rect = page.rect
    clip = pymupdf.Rect(rect.x0, rect.y0 + rect.height * y0 / 1000,
                        rect.x1, rect.y0 + rect.height * y1 / 1000)
    pix = page.get_pixmap(dpi=200, clip=clip)
    gray = pymupdf.Pixmap(pymupdf.csGRAY, pix)
    dark = sum(1 for b in gray.samples if b < 200)
    if dark < INK_MIN_PIXELS:
        return ""
    b64 = base64.b64encode(pix.tobytes("png")).decode()
    body = {"model": OCR_MODEL, "temperature": 0, "max_tokens": 1024,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": "Text Recognition:"}]}]}
    resp = requests.post(OCR_URL, json=body, timeout=180)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    reason = "too long" if len(text) > MAX_STRIP_CHARS else loop_flag(text)
    if reason:
        log.warning("strip skipped (%s, %d chars): %r", reason, len(text), text[:80])
        return ""
    return drop_noise(text)


def assemble_page(page, regions: list):
    """Header strip + layout regions in reading order + footer strip."""
    info = {"regions": len(regions), "header": False, "footer": False, "fallback": False}
    if not regions:  # layout found nothing: read the whole page as one block
        info["fallback"] = True
        return read_strip(page, 0, 1000), info
    regions = sorted(regions, key=lambda r: r.get("index", 0))
    top = min(r["bbox_2d"][1] for r in regions)
    bottom = max(r["bbox_2d"][3] for r in regions)
    parts = []
    if top > TOP_MIN:
        head = read_strip(page, 0, top - 2)
        if head:
            parts.append(head)
            info["header"] = True
    parts.extend(t for t in (clean_region(r.get("content")) for r in regions) if t)
    if bottom < BOTTOM_MAX:
        foot = read_strip(page, bottom + 2, 1000)
        if foot:
            parts.append(foot)
            info["footer"] = True
    return "\n\n".join(parts), info


# ------------------------------------------------------------- GLM-OCR SDK
def run_layout(src: pathlib.Path, outdir: pathlib.Path, label: str = "") -> list:
    cmd = [GLMOCR_BIN, "parse", str(src), "--output", str(outdir),
           "--layout-device", LAYOUT_DEVICE,
           "--set", "pipeline.maas.enabled", "false",
           "--set", "pipeline.ocr_api.api_host", OCR_HOST,
           "--set", "pipeline.ocr_api.api_port", OCR_PORT,
           "--set", "pipeline.ocr_api.model", OCR_MODEL]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t0 = time.time()
    while True:
        try:
            out, err = proc.communicate(timeout=HEARTBEAT_SECONDS)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            if elapsed > CHUNK_TIMEOUT:
                proc.kill()
                proc.communicate()
                raise RuntimeError(f"chunk exceeded the {CHUNK_TIMEOUT // 60} minute limit")
            status = ollama_status()
            log.info("still working on %s (%.0f min). Ollama: %s",
                     label, elapsed / 60, status or "status unavailable")
    if proc.returncode != 0:
        raise RuntimeError("glmocr failed: " + (err or out)[-600:])
    files = [f for f in pathlib.Path(outdir).rglob("*.json")
             if not f.name.endswith("_model.json")]
    if not files:
        raise RuntimeError("glmocr produced no JSON output")
    return json.loads(files[0].read_text(encoding="utf-8"))


def _split_pdf(doc, a: int, b: int, path: pathlib.Path):
    part = pymupdf.open()
    part.insert_pdf(doc, from_page=a, to_page=b - 1)
    part.save(str(path))
    part.close()


def ocr_document(src: pathlib.Path, workdir: pathlib.Path, cache_dir: pathlib.Path, label: str):
    """OCR a file in chunks of CHUNK_PAGES pages, logging progress after each chunk.
    Finished chunks are cached, so a retry after a failure resumes instead of restarting."""
    doc = pymupdf.open(str(src))
    total = len(doc)
    size = CHUNK_PAGES if (CHUNK_PAGES > 0 and src.suffix.lower() == ".pdf") else total
    size = max(size, 1)
    ranges = [(a, min(a + size, total)) for a in range(0, total, size)]
    log.info("%s: %d pages, %d chunk(s) of up to %d", label, total, len(ranges), size)
    cache_dir.mkdir(parents=True, exist_ok=True)
    texts, infos = [], []
    t0, new_pages = time.time(), 0
    for a, b in ranges:
        cf = cache_dir / f"{a:06d}-{b:06d}.json"
        if cf.exists():
            c = json.loads(cf.read_text(encoding="utf-8"))
            ct, cinfo = c["texts"], c["infos"]
            log.info("%s: pages %d-%d reused from an earlier attempt", label, a + 1, b)
        else:
            warm_model(quiet=True)
            if len(ranges) == 1:
                part = src
            else:
                part = workdir / f"chunk_{a:06d}.pdf"
                _split_pdf(doc, a, b, part)
            pages = run_layout(part, workdir / f"out_{a:06d}", f"{label} pages {a + 1}-{b}")
            if len(pages) != b - a:
                raise RuntimeError(f"page mismatch in pages {a + 1}-{b}: chunk has {b - a} pages, "
                                   f"OCR returned {len(pages)}")
            ct, cinfo = [], []
            for i, regions in enumerate(pages):
                text, info = assemble_page(doc[a + i], regions)
                ct.append(text)
                cinfo.append(info)
            cf.write_text(json.dumps({"texts": ct, "infos": cinfo}), encoding="utf-8")
            new_pages += b - a
            if len(ranges) > 1:
                elapsed = time.time() - t0
                left = (total - b) * elapsed / new_pages
                log.info("%s: pages %d-%d of %d done (%d%%), %.1f min elapsed, ~%.1f min left",
                         label, a + 1, b, total, round(100 * b / total), elapsed / 60, left / 60)
        texts += ct
        infos += cinfo
    return "\n\n".join(t for t in texts if t).strip(), total, infos


# --------------------------------------------------------------- paperless
def api(method, path, **kw):
    r = requests.request(method, f"{URL}{path}", headers=HDR, timeout=300, **kw)
    r.raise_for_status()
    return r


def tag_id(name, create=False):
    if not name:
        return None
    res = api("GET", "/api/tags/", params={"name__iexact": name}).json()["results"]
    if res:
        return res[0]["id"]
    return api("POST", "/api/tags/", json={"name": name}).json()["id"] if create else None


def download_original(doc_id):
    r = api("GET", f"/api/documents/{doc_id}/download/", params={"original": "true"})
    ctype = r.headers.get("Content-Type", "").split(";")[0]
    ext = {"application/pdf": ".pdf", "image/png": ".png",
           "image/jpeg": ".jpg"}.get(ctype)
    if not ext:
        raise RuntimeError(f"unsupported file type: {ctype}")
    return r.content, ext


def ocr_server_ok() -> bool:
    try:
        requests.get(f"http://{OCR_HOST}:{OCR_PORT}/v1/models", timeout=10).raise_for_status()
        return True
    except requests.RequestException as e:
        log.warning("OCR server %s:%s not reachable (%s); will retry next poll",
                    OCR_HOST, OCR_PORT, e)
        return False


def ollama_status() -> str:
    """One-line summary of what Ollama has loaded (empty if the server has no /api/ps)."""
    try:
        r = requests.get(f"http://{OCR_HOST}:{OCR_PORT}/api/ps", timeout=5)
        r.raise_for_status()
        parts = []
        for m in r.json().get("models", []):
            size, vram = m.get("size", 0), m.get("size_vram", 0)
            gpu = f"{round(100 * vram / size)}% GPU" if size else "?"
            parts.append(f"{m.get('name')} ({size / 1e9:.1f}GB, {gpu})")
        return ", ".join(parts) if parts else "nothing loaded"
    except Exception:
        return ""


def warm_model(quiet=False):
    """Ask Ollama to load the OCR model and keep it resident for a couple of minutes,
    so the SDK's 30-second startup check never races a cold model load.
    Other OpenAI-compatible servers don't have /api/generate; that is fine, we just move on."""
    status = ollama_status()
    already = OCR_MODEL in status and not status.startswith("nothing")
    if not quiet and status:
        log.info("Ollama has loaded: %s", status)
    if not already:
        log.info("loading %s now (this can take a while if other models are busy)...", OCR_MODEL)
    t0 = time.time()
    try:
        r = requests.post(f"http://{OCR_HOST}:{OCR_PORT}/api/generate",
                          json={"model": OCR_MODEL, "keep_alive": WARM_KEEP_ALIVE},
                          timeout=300)
        if r.status_code == 200:
            if not quiet or time.time() - t0 > 2:
                log.info("model %s ready (%.1fs)", OCR_MODEL, time.time() - t0)
        else:
            log.info("model warm-up skipped (HTTP %s)", r.status_code)
    except requests.RequestException as e:
        log.warning("model warm-up failed (%s); continuing", e)


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def process(doc, ids):
    log.info("doc %s: processing started (%s)", doc["id"], doc.get("title", ""))
    BACKUPS.mkdir(exist_ok=True)
    (BACKUPS / f"{doc['id']}.txt").write_text(doc.get("content") or "", encoding="utf-8")
    data, ext = download_original(doc["id"])
    warm_model()
    cache_dir = CACHE / f"{doc['id']}-{hashlib.sha1(data).hexdigest()[:10]}"
    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / f"doc{ext}"
        src.write_bytes(data)
        text, n_pages, infos = ocr_document(src, pathlib.Path(tmp), cache_dir,
                                            f"doc {doc['id']}")
    if len(text) < n_pages * MIN_CHARS_PER_PAGE:
        shutil.rmtree(cache_dir, ignore_errors=True)  # don't reuse a bad result on retry
        raise ValueError(f"suspiciously little text ({len(text)} chars, {n_pages} pages)")
    tags = [t for t in doc["tags"] if t != ids["trigger"]]
    if ids["next"] and ids["next"] not in tags:
        tags.append(ids["next"])
    api("PATCH", f"/api/documents/{doc['id']}/", json={"content": text, "tags": tags})
    shutil.rmtree(cache_dir, ignore_errors=True)
    log.info("doc %s ok: %d of %d pages read | header strips %d, footer strips %d, "
             "full-page fallbacks %d | %d chars",
             doc["id"], len(infos), n_pages,
             sum(i["header"] for i in infos), sum(i["footer"] for i in infos),
             sum(i["fallback"] for i in infos), len(text))


def give_up(doc, ids):
    for d in CACHE.glob(f"{doc['id']}-*"):
        shutil.rmtree(d, ignore_errors=True)
    tags = [t for t in doc["tags"] if t != ids["trigger"]] + [ids["fail"]]
    api("PATCH", f"/api/documents/{doc['id']}/", json={"tags": tags})


def run_once(ids):
    docs = api("GET", "/api/documents/",
               params={"tags__id__all": ids["trigger"], "page_size": 25}).json()["results"]
    if VERBOSE:
        log.info("poll: %d tagged document(s) waiting", len(docs))
    if not docs:
        return
    if not ocr_server_ok():
        return
    state, ok, failed = load_state(), 0, 0
    for doc in docs:
        key = str(doc["id"])
        try:
            process(doc, ids)
            state.pop(key, None)
            ok += 1
        except (requests.ConnectionError, requests.Timeout) as e:
            log.warning("doc %s: connection problem (%s); will retry next poll", key, e)
            break
        except Exception as e:  # keep the queue moving
            state[key] = state.get(key, 0) + 1
            failed += 1
            log.error("doc %s failed (attempt %d/%d): %s", key, state[key], MAX_ATTEMPTS, e)
            if state[key] >= MAX_ATTEMPTS:
                give_up(doc, ids)
                state.pop(key, None)
        STATE.write_text(json.dumps(state))
    log.info("RUN COMPLETE: %d of %d documents processed, %d failed", ok, len(docs), failed)


def main():
    if not URL or not TOKEN:
        sys.exit("Set PAPERLESS_URL and PAPERLESS_TOKEN first.")
    ids = {"trigger": tag_id(TRIGGER, create=True),
           "next": tag_id(NEXT, create=True),
           "fail": tag_id(FAIL, create=True)}
    log.info("worker started: watching tag '%s', OCR server %s", TRIGGER, OCR_URL)
    while True:
        try:
            run_once(ids)
        except Exception as e:
            log.error("poll failed: %s", e)
        time.sleep(POLL)


if __name__ == "__main__":
    main()
