# paperless-glm-ocr-worker

A small, tag-driven worker that re-OCRs documents in [paperless-ngx](https://docs.paperless-ngx.com/) with
[GLM-OCR](https://github.com/zai-org/GLM-OCR) and writes the text back, with no manual handling.

Tag a document with the trigger tag. The worker downloads the original file, OCRs it, replaces the
document's text, and swaps the trigger tag for a "done" tag (or a "failed" tag after repeated errors).

> **Status:** early-stage. Built for and tested on one setup (Unraid, Ollama on an NVIDIA GPU,
> paperless-ngx in Docker) with a modest number of real documents. Expect rough edges.
> **It overwrites each document's stored text.** The previous text is saved first (see *Backups*).

## Why

GLM-OCR works best through its two-stage pipeline: a layout model finds regions, then a small vision model
reads each region. Generic "send the page to a vision LLM" integrations skip the layout stage. This worker runs
the full pipeline through the `glmocr` SDK, then patches the gaps the layout stage leaves.

## How it works

1. Polls paperless-ngx for documents carrying `TRIGGER_TAG`.
2. Downloads the original file (PDF or image).
3. Splits long PDFs into chunks (default 25 pages) and runs `glmocr parse` on each chunk
   (layout detection on CPU, region OCR on your model server).
4. Reads the **top and bottom strips** of each page separately. The layout model often does not box
   headers, footers, logos or page numbers, so text there would otherwise be lost.
5. Flattens tables to plain `cell | cell` rows and removes known fillable-PDF noise.
6. Writes the text to the document's `content` and swaps the tags.

Other behavior:

- **Progress logging** after each chunk, with an estimate of time left, plus a periodic "still working" line that
  includes what the model server has loaded (and how much of each model is on the GPU).
- **Resume:** finished chunks are cached, so a retry after a failure does not start from page 1.
- **Retries:** up to `MAX_ATTEMPTS` per document, then the failure tag is applied. Connection problems
  (paperless or the model server being down) are not counted as attempts.
- **Backups:** the original text of each document is saved to `BACKUP_DIR/<id>.txt` before it is replaced.
- **Sanity checks:** a document whose text is far too short for its page count counts as a failure.
  Header/footer strips that look like a repetition loop are discarded and logged.
- **No cloud:** the SDK's hosted mode is explicitly disabled. Nothing is sent to the model vendor.

## Requirements

- paperless-ngx and an API token.
- A model server with an OpenAI-compatible `/v1/chat/completions` endpoint serving GLM-OCR.
  Tested with **Ollama** (`glm-ocr:bf16`). The model-load warm-up uses Ollama's `/api/generate`;
  other servers simply skip it.
- Docker (the image installs CPU-only PyTorch for the layout model).

## Quick start (Docker Compose)

1. Pull the model on your Ollama host and create a copy with a smaller context window (important, see *Ollama notes*):

   ```bash
   ollama pull glm-ocr:bf16
   printf 'FROM glm-ocr:bf16\nPARAMETER num_ctx 16384\n' > Modelfile
   ollama create glm-ocr-16k -f Modelfile
   ```

2. Edit `compose.yml`: set `PAPERLESS_URL`, `PAPERLESS_TOKEN`, `OCR_HOST`, and put the container on the same
   Docker network as paperless-ngx and Ollama.
3. Build and start:

   ```bash
   docker compose up -d --build
   docker logs -f glm-worker
   ```

   You should see `worker started: watching tag 'glm-ocr'`.
4. In paperless-ngx, add the trigger tag to a document. It is picked up within `POLL_SECONDS`.

After changing `glm_worker.py` or the Dockerfile you must rebuild (`docker compose up -d --build`);
a plain restart reuses the old image. Changing only environment variables needs no rebuild.

## Settings

All settings are environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `PAPERLESS_URL` | *(required)* | Base URL of paperless-ngx |
| `PAPERLESS_TOKEN` | *(required)* | paperless-ngx API token |
| `TRIGGER_TAG` | `glm-ocr` | Tag that starts OCR |
| `NEXT_TAG` | `glm-ocr-done` | Added on success (empty = none). Use it to trigger other tools |
| `FAIL_TAG` | `glm-ocr-failed` | Added after `MAX_ATTEMPTS` failures |
| `OCR_HOST` / `OCR_PORT` | `localhost` / `11434` | Model server |
| `OCR_MODEL` | `glm-ocr` | Model name as the server knows it |
| `LAYOUT_DEVICE` | `cpu` | Device for the layout model |
| `POLL_SECONDS` | `60` | Polling interval |
| `MAX_ATTEMPTS` | `3` | Tries per document before the failure tag |
| `MIN_CHARS_PER_PAGE` | `40` | Average-text sanity check |
| `CHUNK_PAGES` | `25` | Pages per layout run (`0` = whole file) |
| `CHUNK_TIMEOUT_SECONDS` | `1800` | Time limit per chunk |
| `HEARTBEAT_SECONDS` | `120` | Interval of the "still working" log line |
| `WARM_KEEP_ALIVE` | `2m` | How long the worker asks Ollama to keep the model loaded |
| `VERBOSE` | `0` | `1` logs every idle poll |
| `STATE_FILE`, `BACKUP_DIR`, `CACHE_DIR`, `LOG_FILE` | under `/data` in the compose file | Persistence and logs |
| `TZ` | *(unset = UTC)* | Log timestamps |

## Ollama notes

- **Context size.** Ollama's default context (4096) is too small for page images. A very large context
  also reserves a lot of memory: in testing, a 64000-token context made the 2.7 GB model grow to 6.7 GB and spill
  partly onto the CPU, dropping generation from a fast GPU rate to about 21 tokens/s. A copy created with
  `num_ctx 16384` fit entirely on the GPU. Use the 16K copy for OCR.
- **Cold starts.** The SDK checks the server for 30 seconds on startup. If the model was unloaded, loading it
  can take longer, so the worker loads it first (and waits up to 5 minutes) before every chunk.
- **Sharing the GPU.** Ollama can run several models at once if they fit. If another model is busy or
  VRAM is short, requests can queue, and the startup check can time out. The failed attempt is retried.
- **Keep-alive.** The worker sets its own keep-alive for the OCR model, so you do not need to change the server-wide value.

## paperless-ngx notes

- paperless-ngx rejects requests whose `Host` header is not in its allowed list. If the worker gets `400 Bad Request`
  using an internal container name, either use the public URL, or add the internal name to
  `PAPERLESS_ALLOWED_HOSTS` on the paperless container (keeping existing entries).
- Tag names are matched case-insensitively. Missing tags are created automatically.
- The worker handles one document at a time.

## Known limitations

- **Layout misses.** Text outside every detected box (gaps between regions, side margins), tiny text inside
  dense form boxes, and text inside logos or stamps can be missed. Header and footer strips recover part of this.
- **Handwriting.** Layout detection is trained on printed pages and may box only part of a handwritten page.
  Faded handwriting on carbon copies was missed in testing. Test a sample before running a large archive.
- **Speed.** Plain text runs at a few seconds per page; dense forms can be ten times slower. The layout step runs on
  the CPU. GPU layout has not been verified.
- **Noise filter.** `NOISE` in `glm_worker.py` removes interface text from fillable PDFs ("Click & Tab To Begin",
  "Reset Form"). Edit the list for your own documents.
- **No tests in this repository yet.** Behavior was checked with mocks and by hand.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Failed to connect to API server ... within 30 seconds` | Model was cold or the server was busy; the worker retries. Check `ollama ps` |
| `400 Bad Request` from paperless | Hostname not in `PAPERLESS_ALLOWED_HOSTS` |
| `404` from the model server | `OCR_MODEL` does not match a model name on the server |
| Model shows partly on CPU in `ollama ps` | Context too large or VRAM shared; use the 16K copy |
| Code changes have no effect | The image was not rebuilt |
| Document landed on the failure tag | Fix the cause, then re-tag it with the trigger tag; the attempt count starts fresh |

## License

MIT. This project depends on the [`glmocr`](https://github.com/zai-org/GLM-OCR) SDK and the GLM-OCR model; check their
licenses for your use.
