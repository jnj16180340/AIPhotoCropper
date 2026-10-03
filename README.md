# Cropper

Automated pipeline for cropping scanned photographs using a local vision-language model. Detects the bounding box of actual photo content (excluding scanner borders, margins, timestamps, and artifacts), then crops the original image at full resolution.

Runs against any OpenAI-compatible local model server. Currently tested with
**Qwen3-VL-8B-Instruct (Q8_0)** served by [Lemonade](https://lemonade-server.ai) on the
`vulkan` llama.cpp backend — see [Model runner backend](#model-runner-backend), which
you should read before changing runner, model, or backend. Originally developed against
[LMStudio](https://lmstudio.ai).

---

## How it works

```
Source image (any supported format)
  │
  ▼  generate_preview()
8-bit PNG preview (≤1000 px on longest side)
  │
  ▼  call_model()  →  model server (OpenAI-compatible API)  →  Qwen3-VL
Bounding box in 1000×1000 coordinate space  {x1, y1, x2, y2}
  │
  ▼  scale_bbox_to_original()
Pixel-space bounding box in original image dimensions
  │
  ▼  crop_and_save_image()
Cropped output  (16-bit depth preserved for TIFFs)
```

Qwen3-VL reports bounding boxes as integers in \[0, 1000\], normalised **per axis and
independently** — not against a square grid. Each axis is rescaled back to the original
image dimensions with a simple linear mapping:

```
x_orig = bbox_x * image_width  / 1000
y_orig = bbox_y * image_height / 1000
```

> **Note on letterboxing:** if the model padded rather than stretched to reach its
> internal grid, coordinates on the short axis would carry a padding offset.
> Spot-checked 2026-09-22 against 773×1000 previews, including a rotated photo whose
> box spans the long axis: boxes land correctly under plain per-axis scaling, so no
> offset compensation is needed. If crops ever appear systematically shifted on one
> axis, `scale_bbox_to_original()` in `crop.py` is where to add it.

---

## Requirements

Dependencies are declared in `pyproject.toml` (`numpy`, `openai`, `pillow`, `tifffile`;
Python ≥3.10). With [uv](https://docs.astral.sh/uv/), `uv run crop.py` resolves them on
first run — there is no separate install step. Otherwise `pip install tifffile numpy
Pillow openai`.

A model runner must be serving an OpenAI-compatible API locally with a vision model
loaded. The default base URL is `http://localhost:13305/v1` (Lemonade); override with
`--base-url`. Originally developed against LMStudio on `http://localhost:1234/v1`.

---

## Model runner backend

**On AMD Strix Halo (gfx1151 — Ryzen AI Max+ 395 / Radeon 8060S), run llama.cpp on the
`vulkan` backend, not `rocm`.**

The bundled `llamacpp:rocm` backend silently produces corrupt output on this GPU. It
does not error, does not warn, and does not fail fast — it returns a well-formed HTTP
200 containing garbage. Measured 2026-09-22 with `Qwen3-VL-8B-Instruct-GGUF-Q8_0`:

| Backend | Result over 10 scans | Per-image |
|---------|----------------------|-----------|
| `rocm` | correct for 3 requests, then 256 `?` characters with `finish_reason="length"` on every request until the model is reloaded | 11.3 s |
| `vulkan` | 10/10 correct, deterministic across runs | 2.9 s |

Dense models degenerate into repeated `?`; MoE models (anything `-A3B`, `LFM2.5-8B-A1B`)
return empty strings or punctuation spew. Larger dense models such as
`Qwen3.8-27B-GGUF-UD-Q8_K_XL` fail more subtly, emitting fluent but truncated JSON —
which is easy to misread as a prompting problem. It is not.

Upstream, both open as of 2026-09:

- [lemonade-sdk/lemonade#3610](https://github.com/lemonade-sdk/lemonade/issues/3610) —
  bundled rocm backend produces incorrect output on gfx1151. Recommends Vulkan.
- [ggml-org/llama.cpp#28113](https://github.com/ggml-org/llama.cpp/issues/28113) — MoE
  `MUL_MAT_ID` path garbage on RDNA3.5, regression from PR #27621.

### Setting it

In Lemonade Studio, set the backend per model in the model's settings. `POST
/api/v1/load` with `llamacpp_backend` is a **runtime override only** and reverts on the
next reload. The persisted config lives in
`/opt/var/lib/lemonade/.cache/lemonade/recipe_options.json` (owned by the `lemonade`
user; `sudo systemctl restart lemond` after editing).

To check what is actually loaded right now:

```bash
ps aux | grep llama-server | grep -oE '(vulkan|rocm)/llama-server'
```

Note that `--workers N` makes this worse, not better: concurrent requests spread across
llama-server's four slots, and the corruption is triggered once the fourth slot is
touched. On a suspect backend, run with `--workers 1`.

### When to re-evaluate this decision

Vulkan is currently both correct *and* faster here, so there is no standing reason to
revisit. Reconsider only if one of these becomes true:

1. **Both upstream issues close** and Lemonade ships a llama.cpp build that includes the
   fixes. Check the bundled version with `curl -s localhost:13305/api/v1/system-info`
   and compare against the fix commits, rather than trusting the release notes.
2. **Vulkan becomes the bottleneck.** At ~2.9 s/image the model is not the slow part of
   this pipeline; TIFF I/O is. Only chase ROCm's prefill throughput if profiling shows
   inference actually dominates a batch.
3. **A Vulkan-specific correctness bug appears** — i.e. this same failure mode shows up
   on Vulkan too. Then it is a backend bake-off again, not a settled question.
4. **The hardware changes.** This is a gfx1151 defect. On a discrete AMD card, or a
   different GPU vendor entirely, none of the above applies and ROCm/CUDA is fine.

Do **not** re-evaluate on the strength of a few good responses. Corruption here begins
at the fourth request and a short smoke test will pass on a broken backend. The
regression check is:

```bash
# Expect 10/10 parsed, zero retries, and no "did not finish cleanly" errors.
uv run crop.py --input-dir <a directory of 10+ scans> --bbox-only --workers 1
```

Coordinates should also be *stable* between two consecutive runs of that command. A
backend that parses cleanly but returns different boxes each time is still broken;
`crop.py` sends `temperature=0.0`, so identical input must produce identical output.

---

## Supported input formats

| Format | Extension | Notes |
|--------|-----------|-------|
| TIFF | `.tif`, `.tiff` | 16-bit depth preserved on crop |
| PNG | `.png` | including 16-bit PNG |
| JPEG | `.jpg`, `.jpeg` | |

---

## Usage

### Basic

```bash
# All images in the current directory → output/
uv run crop.py

# Specific files
uv run crop.py scan1.tif scan2.tif

# Directory or glob
uv run crop.py scans/
uv run crop.py 'batch_*/scan*.tif'
```

### Directories

With no positional arguments, `--input-dir` decides what gets processed. Previews and
output default to subdirectories of it, and either can be redirected:

```bash
# Read one scan batch, write crops somewhere else entirely
uv run crop.py --input-dir '/scans/64 Blaisdell Drive - 1987' --output-dir ~/Pictures/TiffScans

# Keep previews out of the source directory
uv run crop.py --input-dir scans/ --preview-dir /tmp/previews
```

| Flag | Default |
|------|---------|
| `--input-dir` | `.` — scanned one level deep unless `--recursive`, and only when no `PATH` args are given |
| `--output-dir` | `<input-dir>/output` |
| `--preview-dir` | `<input-dir>/previews` |

`crop_log.json` is always written to `--input-dir`, not to `--output-dir`. Any file whose
crop already exists at its computed destination is skipped unless `--force` is passed.

### Recursive scanning

`--recursive` (`-r`) walks the whole subtree of a directory input instead of one level,
and **the output and preview trees mirror the input structure**:

```bash
uv run crop.py --input-dir scans/ --recursive
uv run crop.py 'scans/**/*.tif'        # ** globs already recurse; no flag needed
```

```
scans/                          scans/output/                scans/previews/
├── top.tif                     ├── top.tif                  ├── top.png
├── 1987/roll-a/scan01.tif  →   ├── 1987/roll-a/scan01.tif   ├── 1987/roll-a/scan01.png
├── 1987/roll-b/scan01.tif      ├── 1987/roll-b/scan01.tif   ├── 1987/roll-b/scan01.png
└── loose/scan02.tif            └── loose/scan02.tif         └── loose/scan02.png
```

The two `scan01.tif` files keep their own outputs. Without mirroring they would
overwrite each other — in the crop, in the preview, *and* in `crop_log.json` — so log
entries are keyed by the path relative to `--input-dir` rather than by bare filename.

Paths are mirrored relative to `--input-dir` whether or not `--recursive` is passed, so
a `**` glob preserves structure too. Files resolving outside `--input-dir` have no
relative position and land flat in the output directory. `--suffix` and `--in-place`
write next to the source, so they preserve structure inherently and are unaffected.

Recursion skips three things it would otherwise consume: `--output-dir` and
`--preview-dir` (both would otherwise be walked as ordinary subdirectories full of
images), and `*.orig.*` backups left by `--in-place`. A recursive run is therefore
re-runnable — the second run finds the same files, not its own output.

Old flat logs stay compatible: for a non-nested layout the relative path *is* the
filename, so `--from-log` keeps working against logs written before this change.

### Output destination

Three modes are mutually exclusive:

```bash
# Default: write to output/ (preserves originals)
uv run crop.py

# Alongside source with a suffix (e.g. scan.tif → scan_crop.tif)
uv run crop.py --suffix _crop

# Overwrite source in place (originals backed up to .orig.ext)
uv run crop.py --in-place
uv run crop.py --in-place --no-backup   # skip backup (dangerous)
```

### Output format

```bash
# Save crops as JPEG regardless of input format
uv run crop.py --output-format jpg

# Save as PNG
uv run crop.py --output-format png

# (not compatible with --in-place)
```

### Crop adjustment

```bash
# Expand the detected bbox by 15 pixels on each side
uv run crop.py --padding 15

# Reject detections that cover less than 10% or more than 95% of the image
uv run crop.py --min-coverage 0.10 --max-coverage 0.95
```

### Inspection and dry runs

```bash
# Generate preview PNGs only — no model call, no crop
uv run crop.py --preview-only

# Call the model and generate annotated previews, but write no output files
uv run crop.py --dry-run

# Print detected bboxes to stdout without writing anything
uv run crop.py --bbox-only
```

After every successful model call an annotated preview is written to
`previews/<stem>_bbox.png` with the predicted crop rectangle drawn in red.
This is the fastest way to check whether the model is detecting the right area.

### Re-running without the model

```bash
# Re-apply crops from crop_log.json (no API call)
# Useful for re-running with different --padding or --output-format
uv run crop.py --from-log --force --padding 20
uv run crop.py --from-log --output-format jpg --force
```

### Performance and reliability

```bash
# Parallel workers (I/O and API calls are parallelised)
uv run crop.py --workers 4

# Retry the model up to 3 times on a failed response
uv run crop.py --retries 3

# Reprocess files that already have output
uv run crop.py --force
```

### Custom prompt

```bash
# Use a custom prompt file instead of the built-in one
uv run crop.py --prompt my_prompt.txt
```

The default prompt asks the model to find the photo content and return a JSON bounding box. A custom prompt file can target different content (e.g. documents, stamps, faces) without modifying the script.

`call_model()` rejects any completion whose `finish_reason` is not `"stop"` before
parsing is attempted, so a truncated or degenerate response reports itself as
`Model did not finish cleanly (finish_reason='length', ...)` rather than as a bounding
box parse failure. If you see that error, suspect the runner, not the prompt — see
[Model runner backend](#model-runner-backend).

### Model and API

```bash
# Override the model name (must match the id the server reports, exactly)
uv run crop.py --model "Qwen3-VL-8B-Instruct-GGUF-Q8_0"

# Override the API base URL (default: http://localhost:13305/v1)
uv run crop.py --base-url http://192.168.1.10:1234/v1
```

`--model` must match the server's model id exactly. To list what is available:

```bash
curl -s localhost:13305/v1/models | python3 -c 'import json,sys; [print(m["id"]) for m in json.load(sys.stdin)["data"]]'
```

If the server requires an API key, set `LMSTUDIO_API_KEY` (it defaults to the literal
string `lmstudio`, which most local servers ignore):

```bash
LMSTUDIO_API_KEY=... uv run crop.py --input-dir scans/
```

---

## Output files

| Path | Description |
|------|-------------|
| `output/<filename>` | Cropped image (default mode) |
| `previews/<stem>.png` | 8-bit downsampled preview sent to the model |
| `previews/<stem>_bbox.png` | Preview with predicted crop box overlaid in red |
| `crop_log.json` | Per-run log of all results (bbox coords, model response, status) |
| `<stem>.orig.<ext>` | Backup of original file when using `--in-place` |

`crop_log.json` is appended after each file is processed, so a run that is interrupted mid-batch leaves a valid partial log. `--from-log` reads this file to re-apply crops without calling the model again.

---

## Known limitations and future work

- **TIFF metadata:** `tifffile.imwrite` does not copy EXIF/XMP tags from the source. Add metadata passthrough if archival fidelity of metadata is required.

- **Interactive review:** a `--review` flag that opens each `_bbox.png` and prompts y/n before writing the crop would be useful for quality-checking batches without a separate viewer.

- **Worker concurrency:** `--workers N` submits N concurrent requests to the model runner. The right value depends on the host machine; start with 2–4 and increase if the server handles it. See the backend note above before raising it on gfx1151.

- **`--image-min-tokens` is not worth setting.** llama.cpp warns at load that Qwen-VL
  wants ≥1024 image tokens for grounding tasks, and a 773×1000 preview produces ~860.
  Tested 2026-09-22: forcing 1024 moved bounding boxes by under 0.5% and cost ~25% more
  latency (2.9 s → 3.7 s per image). The ≤1000 px preview is already past the point of
  diminishing returns for this task.

---

## Change log

| Date | Change |
|------|--------|
| 2026-02-20 | Initial scaffold |
| 2026-02-23 | Input globbing and directory support |
| 2026-02-28 | Multi-format input; output modes (in-place/suffix); padding; annotated previews; coverage sanity check; retries; parallel workers; custom prompt; bbox-only; from-log |
| 2026-09-22 | `call_model()` checks `finish_reason`; `test_crop.py` self-check; documented the gfx1151 ROCm corruption and the Vulkan requirement |
| 2026-09-22 | `--recursive`; output/preview trees mirror the input structure; log keyed by relative path; recursion skips output/, previews/ and `.orig` backups |
