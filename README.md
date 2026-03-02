# Cropper

Automated pipeline for cropping scanned photographs using a local vision-language model. Detects the bounding box of actual photo content (excluding scanner borders, margins, timestamps, and artifacts), then crops the original image at full resolution.

Tested with **qwen3-vl-8b-instruct** via [LMStudio](https://lmstudio.ai).

---

## How it works

```
Source image (any supported format)
  │
  ▼  generate_preview()
8-bit PNG preview (≤1000 px on longest side)
  │
  ▼  call_model()  →  LMStudio (OpenAI-compatible API)  →  qwen3-vl
Bounding box in 1000×1000 coordinate space  {x1, y1, x2, y2}
  │
  ▼  scale_bbox_to_original()
Pixel-space bounding box in original image dimensions
  │
  ▼  crop_and_save_image()
Cropped output  (16-bit depth preserved for TIFFs)
```

qwen3-vl internally normalises all images to a 1000×1000 grid and reports bounding boxes as integers in \[0, 1000\]. These are rescaled back to the original image dimensions with a simple linear mapping:

```
x_orig = bbox_x * image_width  / 1000
y_orig = bbox_y * image_height / 1000
```

> **Note on letterboxing:** if qwen3-vl pads rather than stretches images to fit 1000×1000, coordinates on the short axis will include a padding offset. In practice with the 8b model this has not been an issue, but if crops appear systematically shifted on one axis, `scale_bbox_to_original()` in `crop.py` is where to add offset compensation.

---

## Requirements

```bash
pip install tifffile numpy Pillow openai
```

LMStudio must be running locally with a vision model loaded (e.g. `qwen3-vl-8b-instruct`). The API is accessed at `http://localhost:1234/v1` by default.

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
python crop.py

# Specific files
python crop.py scan1.tif scan2.tif

# Directory or glob
python crop.py scans/
python crop.py 'batch_*/scan*.tif'
```

### Output destination

Three modes are mutually exclusive:

```bash
# Default: write to output/ (preserves originals)
python crop.py

# Alongside source with a suffix (e.g. scan.tif → scan_crop.tif)
python crop.py --suffix _crop

# Overwrite source in place (originals backed up to .orig.ext)
python crop.py --in-place
python crop.py --in-place --no-backup   # skip backup (dangerous)
```

### Output format

```bash
# Save crops as JPEG regardless of input format
python crop.py --output-format jpg

# Save as PNG
python crop.py --output-format png

# (not compatible with --in-place)
```

### Crop adjustment

```bash
# Expand the detected bbox by 15 pixels on each side
python crop.py --padding 15

# Reject detections that cover less than 10% or more than 95% of the image
python crop.py --min-coverage 0.10 --max-coverage 0.95
```

### Inspection and dry runs

```bash
# Generate preview PNGs only — no model call, no crop
python crop.py --preview-only

# Call the model and generate annotated previews, but write no output files
python crop.py --dry-run

# Print detected bboxes to stdout without writing anything
python crop.py --bbox-only
```

After every successful model call an annotated preview is written to
`previews/<stem>_bbox.png` with the predicted crop rectangle drawn in red.
This is the fastest way to check whether the model is detecting the right area.

### Re-running without the model

```bash
# Re-apply crops from crop_log.json (no API call)
# Useful for re-running with different --padding or --output-format
python crop.py --from-log --force --padding 20
python crop.py --from-log --output-format jpg --force
```

### Performance and reliability

```bash
# Parallel workers (I/O and API calls are parallelised)
python crop.py --workers 4

# Retry the model up to 3 times on parse failure
python crop.py --retries 3

# Reprocess files that already have output
python crop.py --force
```

### Custom prompt

```bash
# Use a custom prompt file instead of the built-in one
python crop.py --prompt my_prompt.txt
```

The default prompt asks the model to find the photo content and return a JSON bounding box. A custom prompt file can target different content (e.g. documents, stamps, faces) without modifying the script.

### Model and API

```bash
# Override the model name (must match the name shown in LMStudio)
python crop.py --model "qwen3-vl-8b-instruct"

# Override the API base URL
python crop.py --base-url http://192.168.1.10:1234/v1
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

- **Recursive scanning:** directory inputs are scanned one level deep. Pass explicit glob patterns with `**` to recurse (e.g. `'scans/**/*.tif'`). A `--recursive` flag would be a natural addition.

- **Interactive review:** a `--review` flag that opens each `_bbox.png` and prompts y/n before writing the crop would be useful for quality-checking batches without a separate viewer.

- **Worker concurrency:** `--workers N` submits N concurrent requests to LMStudio. The right value depends on the host machine; start with 2–4 and increase if the server handles it.

---

## Change log

| Date | Change |
|------|--------|
| 2026-02-20 | Initial scaffold |
| 2026-02-23 | Input globbing and directory support |
| 2026-02-28 | Multi-format input; output modes (in-place/suffix); padding; annotated previews; coverage sanity check; retries; parallel workers; custom prompt; bbox-only; from-log |
