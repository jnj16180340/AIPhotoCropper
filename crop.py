#!/usr/bin/env python3
"""
Automated photo cropping using LMStudio API (qwen3-vl model).

Pipeline:
  1. Downsample source image → 8-bit PNG preview for the LLM
  2. Send PNG to qwen3-vl via LMStudio (OpenAI-compatible) API
  3. Model reports a bounding box in 1000×1000 coordinate space
  4. Rescale bounding box to original image pixel dimensions
  5. Crop original image (preserving bit depth) → save to configured destination

Supported input formats: TIFF (.tif/.tiff), PNG (.png), JPEG (.jpg/.jpeg)

Output destination modes (mutually exclusive):
  default          write to --output-dir (default: ./output/)
  --suffix _crop   write alongside source as filename_crop.ext
  --in-place       overwrite source (backs up to filename.orig.ext unless --no-backup)

Usage:
  python crop.py                              # all images in current dir
  python crop.py scans/ '*.tif'              # dir or glob
  python crop.py --in-place *.tif            # overwrite originals (backed up)
  python crop.py --suffix _crop *.tif        # write alongside source
  python crop.py --padding 20                # expand crop by 20 px each side
  python crop.py --output-format jpg         # save crops as JPEG
  python crop.py --bbox-only                 # print bboxes only, write nothing
  python crop.py --from-log                  # re-apply crops from log (no API call)
  python crop.py --workers 4                 # parallel processing
  python crop.py --prompt my_prompt.txt      # custom prompt file
  python crop.py --preview-only              # generate previews only
  python crop.py --dry-run                   # previews + no crop/API call

Requires: tifffile, numpy, Pillow, openai
  pip install tifffile numpy Pillow openai
"""

import argparse
import base64
import glob as _glob
import json
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
from openai import OpenAI
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"
DEFAULT_MODEL = "qwen3-vl-8b-instruct"
QWEN_COORD_SIZE = 1000          # model reports boxes in 1000×1000 space
PREVIEW_MAX_PX = 1000           # longest side of the generated PNG preview
DEFAULT_RETRIES = 2
DEFAULT_MIN_COVERAGE = 0.05     # bbox must cover ≥5% of image area
DEFAULT_MAX_COVERAGE = 0.99     # bbox must cover ≤99% of image area

SYSTEM_PROMPT = (
    "You are a precise image analysis tool. "
    "Return only valid JSON — no explanation, no markdown."
)

DEFAULT_USER_PROMPT = """\
This image is a scan that may contain white/grey borders, scanner artifacts,
timestamps, or margin areas surrounding the actual photographic content.

Identify the tight bounding box around the main photo content (the real
image area, excluding borders and artifacts).

Return ONLY a JSON object with integer coordinates in the 0–1000 range:
{"x1": <left>, "y1": <top>, "x2": <right>, "y2": <bottom>}"""

OUTPUT_DIR_NAME = "output"
PREVIEW_DIR_NAME = "previews"
LOG_FILE_NAME = "crop_log.json"

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
TIFF_EXTS   = {".tif", ".tiff"}

# PIL format name keyed by output extension
_PIL_FMT = {
    ".tif": "TIFF", ".tiff": "TIFF",
    ".png": "PNG",
    ".jpg": "JPEG", ".jpeg": "JPEG",
}

# ---------------------------------------------------------------------------
# Image I/O  (format-agnostic)
# ---------------------------------------------------------------------------


def load_image_array(path: Path) -> np.ndarray:
    """Load any supported image as a numpy array, preserving bit depth."""
    if path.suffix.lower() in TIFF_EXTS:
        return tifffile.imread(str(path))
    img = Image.open(path)
    return np.array(img)


def array_to_8bit(arr: np.ndarray) -> np.ndarray:
    """Safely convert any integer/float array to uint8."""
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr >> 8).astype(np.uint8)
    arr_f = arr.astype(np.float32)
    lo, hi = arr_f.min(), arr_f.max()
    if hi > lo:
        arr_f = (arr_f - lo) / (hi - lo) * 255.0
    return arr_f.astype(np.uint8)


def generate_preview(src_path: Path, preview_path: Path,
                     max_px: int = PREVIEW_MAX_PX) -> tuple[int, int]:
    """
    Downsample any supported image to an 8-bit PNG preview.
    Returns (original_width, original_height).
    """
    arr = load_image_array(src_path)
    orig_h, orig_w = arr.shape[:2]
    img = Image.fromarray(array_to_8bit(arr))
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(preview_path), format="PNG", optimize=False)
    print(f"  Preview: {orig_w}×{orig_h} → {img.width}×{img.height} px  →  {preview_path.name}")
    return orig_w, orig_h


def annotate_preview(preview_path: Path, bbox_1000: tuple[int, int, int, int],
                     out_path: Path) -> None:
    """Draw the predicted bbox on a copy of the preview PNG (red rectangle)."""
    img = Image.open(preview_path).convert("RGB")
    w, h = img.size
    sx, sy = w / QWEN_COORD_SIZE, h / QWEN_COORD_SIZE
    x1, y1, x2, y2 = (int(bbox_1000[0]*sx), int(bbox_1000[1]*sy),
                       int(bbox_1000[2]*sx), int(bbox_1000[3]*sy))
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="PNG")


def crop_and_save_image(read_path: Path, output_path: Path,
                        bbox: tuple[int, int, int, int],
                        output_format: str | None = None) -> None:
    """
    Crop an image using pixel-space bbox (x1, y1, x2, y2) and save.

    - TIFFs cropped via tifffile to preserve 16-bit depth.
    - All other formats (and format conversions) use PIL.
    - output_format overrides the format inferred from output_path's extension,
      e.g. pass 'JPEG' to save a TIFF crop as JPEG.
    """
    x1, y1, x2, y2 = bbox
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ext = output_path.suffix.lower()
    fmt = output_format or _PIL_FMT.get(ext, "PNG")

    if read_path.suffix.lower() in TIFF_EXTS and fmt == "TIFF":
        arr = tifffile.imread(str(read_path))
        cropped = arr[y1:y2, x1:x2]
        if cropped.ndim == 3 and cropped.shape[2] == 3:
            ph = "rgb"
        elif cropped.ndim == 3 and cropped.shape[2] == 4:
            ph = "rgba"
        else:
            ph = None
        kwargs = dict(photometric=ph) if ph else {}
        tifffile.imwrite(str(output_path), cropped, **kwargs)
        out_w, out_h = cropped.shape[1], cropped.shape[0]
    else:
        img = Image.open(read_path)
        cropped = img.crop((x1, y1, x2, y2))
        save_kw: dict = {}
        if fmt == "JPEG":
            save_kw["quality"] = 95
            if cropped.mode == "RGBA":
                cropped = cropped.convert("RGB")
        cropped.save(str(output_path), format=fmt, **save_kw)
        out_w, out_h = cropped.width, cropped.height

    mb = output_path.stat().st_size / 1_048_576
    print(f"  Saved:  {output_path.name}  ({out_w}×{out_h} px, {mb:.1f} MB)")


# ---------------------------------------------------------------------------
# LMStudio / model interaction
# ---------------------------------------------------------------------------


def image_to_data_url(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def call_model(client: OpenAI, model: str,
               preview_path: Path, user_prompt: str) -> str:
    """Send preview image to the model and return the raw text response."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": image_to_data_url(preview_path)}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        max_tokens=256,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Bounding-box parsing, validation, and scaling
# ---------------------------------------------------------------------------


def parse_bbox_from_response(text: str) -> tuple[int, int, int, int]:
    """
    Extract (x1, y1, x2, y2) from model response.

    Handles:
      • Plain JSON:          {"x1":10,"y1":20,"x2":800,"y2":950}
      • Markdown-fenced JSON
      • Qwen box tags:       <|box_start|>(x1,y1),(x2,y2)<|box_end|>
      • Bare coordinate pairs: (x1,y1),(x2,y2)
    """
    clean = re.sub(r"```[a-z]*\n?", "", text).strip()

    json_match = re.search(r"\{[^}]+\}", clean, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            return (int(obj["x1"]), int(obj["y1"]),
                    int(obj["x2"]), int(obj["y2"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    coord_match = re.search(
        r"\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)", clean)
    if coord_match:
        return tuple(map(int, coord_match.groups()))

    raise ValueError(f"Could not parse bounding box from model response:\n{text}")


def validate_bbox(bbox: tuple[int, int, int, int],
                  coord_size: int = QWEN_COORD_SIZE) -> tuple[int, int, int, int]:
    """Clamp bbox to [0, coord_size] and verify it is non-degenerate."""
    x1, y1, x2, y2 = (max(0, min(v, coord_size)) for v in bbox)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate bbox after clamping: {(x1,y1,x2,y2)}")
    return x1, y1, x2, y2


def check_coverage(bbox: tuple[int, int, int, int],
                   min_cov: float, max_cov: float) -> float:
    """
    Raise ValueError if the bbox covers an implausible fraction of the
    1000×1000 image area.  Returns the coverage ratio on success.
    """
    x1, y1, x2, y2 = bbox
    ratio = (x2 - x1) * (y2 - y1) / QWEN_COORD_SIZE ** 2
    if ratio < min_cov:
        raise ValueError(
            f"Bbox too small: covers {ratio:.1%} of image (threshold {min_cov:.1%})")
    if ratio > max_cov:
        raise ValueError(
            f"Bbox too large: covers {ratio:.1%} of image (threshold {max_cov:.1%})")
    return ratio


def apply_padding(bbox: tuple[int, int, int, int],
                  orig_w: int, orig_h: int, padding: int) -> tuple[int, int, int, int]:
    """Expand bbox by *padding* pixels on each side, clamped to image bounds."""
    x1, y1, x2, y2 = bbox
    return (max(0, x1 - padding), max(0, y1 - padding),
            min(orig_w, x2 + padding), min(orig_h, y2 + padding))


def scale_bbox_to_original(bbox_1000: tuple[int, int, int, int],
                            orig_w: int, orig_h: int) -> tuple[int, int, int, int]:
    """
    Scale a bbox from qwen3-vl's 1000×1000 coordinate space to the
    original image's pixel dimensions via simple linear mapping.

    Note: if the model letterboxes rather than stretches, coordinates on the
    short axis will include padding offsets — adjust this function if crops
    appear systematically off-centre on one axis.
    """
    sx, sy = orig_w / QWEN_COORD_SIZE, orig_h / QWEN_COORD_SIZE
    x1, y1 = int(bbox_1000[0] * sx), int(bbox_1000[1] * sy)
    x2, y2 = int(bbox_1000[2] * sx), int(bbox_1000[3] * sy)
    return (max(0, min(x1, orig_w)), max(0, min(y1, orig_h)),
            max(0, min(x2, orig_w)), max(0, min(y2, orig_h)))


# ---------------------------------------------------------------------------
# Output destination helpers
# ---------------------------------------------------------------------------

_FMT_EXT = {"tif": ".tif", "tiff": ".tiff",
             "png": ".png", "jpg": ".jpg", "jpeg": ".jpg"}


def compute_output_path(src_path: Path, args, output_dir: Path) -> Path:
    """
    Return the destination path for a crop, based on the active output mode.

    Modes (mutually exclusive via CLI validation):
      --in-place  →  same path as source (same extension)
      --suffix S  →  <src_dir>/<stem><S><ext>
      default     →  <output_dir>/<stem><ext>

    If --output-format is set, the extension (and thus format) is overridden,
    except when --in-place is active (format conversion in-place is disallowed).
    """
    ext = (src_path.suffix if args.in_place
           else _FMT_EXT.get(getattr(args, "output_format", None) or "", src_path.suffix))
    if args.in_place:
        return src_path
    if args.suffix:
        return src_path.parent / f"{src_path.stem}{args.suffix}{ext}"
    return output_dir / f"{src_path.stem}{ext}"


def backup_original(path: Path) -> Path:
    """
    Rename *path* to <stem>.orig<ext> (atomic on the same filesystem).
    Returns the backup path.  Raises FileExistsError if backup already exists.
    """
    backup = path.with_name(f"{path.stem}.orig{path.suffix}")
    if backup.exists():
        raise FileExistsError(f"Backup already exists: {backup}  (use --force to overwrite)")
    path.rename(backup)
    return backup


def already_processed(src_path: Path, output_path: Path, in_place: bool) -> bool:
    """Heuristic: True if this file appears to have been processed already."""
    if in_place:
        # A backup file is the only evidence when overwriting in place.
        # Note: if --no-backup was used previously, this will return False
        # even though the file was already processed.
        return src_path.with_name(f"{src_path.stem}.orig{src_path.suffix}").exists()
    return output_path.exists()


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------


def load_log(log_path: Path) -> list:
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)
    return []


def save_log(log_path: Path, entries: list) -> None:
    with open(log_path, "w") as f:
        json.dump(entries, f, indent=2)


def build_log_cache(log_path: Path) -> dict[str, list[int]]:
    """
    Return {filename: bbox_orig} from the most recent 'ok' entry per file.
    Used by --from-log to skip the model API call.
    """
    cache: dict[str, list[int]] = {}
    for entry in load_log(log_path):
        if entry.get("status") == "ok" and entry.get("bbox_orig"):
            cache[entry["file"]] = entry["bbox_orig"]
    return cache


# ---------------------------------------------------------------------------
# Input collection
# ---------------------------------------------------------------------------


def collect_image_paths(inputs: list[str], output_dir: Path | None,
                        exclude_suffix: str | None = None) -> list[Path]:
    """
    Resolve a mixed list of files, directories, and glob patterns into a
    deduplicated, sorted list of image paths.

    Each item is tried in order:
      1. Existing directory → all image files directly inside it
      2. Existing file      → use directly (if a supported image format)
      3. Otherwise          → treat as a shell glob pattern (** supported)
    """
    seen: set[Path] = set()
    paths: list[Path] = []

    for item in inputs:
        p = Path(item)

        if p.is_dir():
            candidates = sorted(
                c for c in p.iterdir()
                if c.is_file() and c.suffix.lower() in IMAGE_EXTS
            )
        elif p.exists():
            candidates = [p]
        else:
            expanded = [Path(g) for g in sorted(_glob.glob(item, recursive=True))]
            candidates = [g for g in expanded if g.suffix.lower() in IMAGE_EXTS]
            if not candidates:
                print(f"WARNING: no image files matched: {item!r}", file=sys.stderr)
                continue

        for c in candidates:
            c = c.resolve()
            if c.suffix.lower() not in IMAGE_EXTS:
                continue
            if output_dir and c.parent == output_dir:
                continue
            if exclude_suffix and c.stem.endswith(exclude_suffix):
                continue
            if c not in seen:
                seen.add(c)
                paths.append(c)

    return paths


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------


def process_image(
    src_path: Path,
    output_path: Path,
    preview_dir: Path,
    client: OpenAI | None,
    model: str,
    user_prompt: str,
    *,
    dry_run: bool,
    force: bool,
    padding: int,
    in_place: bool,
    no_backup: bool,
    output_format: str | None,
    min_coverage: float,
    max_coverage: float,
    retries: int,
    bbox_only: bool,
    cached_bbox: list[int] | None,
) -> dict:
    """
    Full pipeline for one image.  Returns a log entry dict.

    If *cached_bbox* is provided (from --from-log), the model API call is
    skipped and the cached bbox_orig is used directly.
    """
    stem = src_path.stem
    preview_path = preview_dir / f"{stem}.png"
    annotated_path = preview_dir / f"{stem}_bbox.png"

    print(f"\n{'='*60}")
    print(f"Processing: {src_path.name}")

    # --- Skip check ---
    if not force and already_processed(src_path, output_path, in_place):
        print("  Skipping (already processed). Use --force to reprocess.")
        return {"file": src_path.name, "status": "skipped"}

    # --- Step 1: Generate preview ---
    print("  Generating preview PNG…")
    orig_w, orig_h = generate_preview(src_path, preview_path)

    if dry_run:
        print("  [dry-run] Stopping before model call.")
        return {"file": src_path.name, "status": "dry-run",
                "original_size": [orig_w, orig_h]}

    # --- Step 2: Determine bbox ---
    raw_response: str
    bbox_1000: tuple[int, int, int, int] | None
    bbox_orig: tuple[int, int, int, int]

    if cached_bbox is not None:
        bbox_orig = tuple(cached_bbox)
        bbox_1000 = None
        raw_response = "(from log)"
        print(f"  Using cached bbox: {bbox_orig}")
    else:
        # Call model with retries
        print(f"  Calling {model} via LMStudio…")
        last_exc: Exception | None = None
        raw_response = ""
        bbox_1000 = None

        for attempt in range(retries + 1):
            if attempt > 0:
                print(f"  Retry {attempt}/{retries}…")
            try:
                raw_response = call_model(client, model, preview_path, user_prompt)
                print(f"  Model response: {raw_response}")
                bbox_1000 = parse_bbox_from_response(raw_response)
                bbox_1000 = validate_bbox(bbox_1000)
                break
            except Exception as exc:
                last_exc = exc
                print(f"  Attempt {attempt+1} failed: {exc}", file=sys.stderr)
        else:
            raise RuntimeError(
                f"All {retries+1} attempt(s) failed. Last: {last_exc}")

        # Coverage sanity check
        coverage = check_coverage(bbox_1000, min_coverage, max_coverage)
        print(f"  Bbox (1000×1000): {bbox_1000}  coverage={coverage:.1%}")

        # Annotate preview with bbox
        annotate_preview(preview_path, bbox_1000, annotated_path)
        print(f"  Annotated preview: {annotated_path.name}")

        # Scale to original pixel space
        bbox_orig = scale_bbox_to_original(bbox_1000, orig_w, orig_h)
        print(f"  Bbox (original px): {bbox_orig}")

    # --- Step 3: Padding ---
    if padding > 0:
        bbox_orig = apply_padding(bbox_orig, orig_w, orig_h, padding)
        print(f"  Bbox after padding: {bbox_orig}")

    # --- bbox-only mode: stop here ---
    if bbox_only:
        print(f"  [bbox-only] not writing output.")
        return {
            "file": src_path.name,
            "status": "bbox-only",
            "original_size": [orig_w, orig_h],
            "bbox_1000": list(bbox_1000) if bbox_1000 else None,
            "bbox_orig": list(bbox_orig),
            "model_response": raw_response,
        }

    # --- Step 4: Backup (in-place only) ---
    read_path = src_path
    if in_place and not no_backup:
        backup = backup_original(src_path)
        read_path = backup
        print(f"  Backed up original → {backup.name}")

    # --- Step 5: Crop and save ---
    print("  Cropping and saving…")
    fmt = _PIL_FMT.get(f".{output_format}", None) if output_format else None
    crop_and_save_image(read_path, output_path, bbox_orig, fmt)

    return {
        "file": src_path.name,
        "status": "ok",
        "original_size": [orig_w, orig_h],
        "bbox_1000": list(bbox_1000) if bbox_1000 else None,
        "bbox_orig": list(bbox_orig),
        "model_response": raw_response,
        "output": str(output_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Crop scanned photos using a VLM via LMStudio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Positional ---
    p.add_argument("files", nargs="*", metavar="PATH",
                   help=(
                       "Files, directories, or glob patterns to process "
                       "(e.g. scans/ '*.tif' 'batch_*/scan*.tiff'). "
                       "Default: all images in --input-dir."
                   ))

    # --- Directories ---
    p.add_argument("--input-dir", type=Path, default=Path("."),
                   help="Directory to scan when no positional args given (default: .)")
    p.add_argument("--preview-dir", type=Path, default=None,
                   help=f"Where to write preview PNGs (default: <input-dir>/{PREVIEW_DIR_NAME})")

    # --- Output destination (mutually exclusive) ---
    dest = p.add_mutually_exclusive_group()
    dest.add_argument("--output-dir", type=Path, default=None,
                      help=f"Output directory (default: <input-dir>/{OUTPUT_DIR_NAME})")
    dest.add_argument("--suffix", metavar="SUFFIX",
                      help="Write crop alongside source with this suffix, e.g. _crop")
    dest.add_argument("--in-place", action="store_true",
                      help="Overwrite source file (backs up to .orig unless --no-backup)")

    p.add_argument("--no-backup", action="store_true",
                   help="With --in-place: skip creating a .orig backup (dangerous)")

    # --- Format ---
    p.add_argument("--output-format", choices=["tif", "tiff", "png", "jpg", "jpeg"],
                   metavar="FMT",
                   help="Output format: tif, png, jpg (default: same as input). "
                        "Ignored with --in-place.")

    # --- Crop tweaks ---
    p.add_argument("--padding", type=int, default=0, metavar="PX",
                   help="Expand detected bbox by PX pixels on each side (default: 0)")
    p.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                   metavar="RATIO",
                   help=f"Reject bbox if it covers less than this fraction of the image "
                        f"(default: {DEFAULT_MIN_COVERAGE})")
    p.add_argument("--max-coverage", type=float, default=DEFAULT_MAX_COVERAGE,
                   metavar="RATIO",
                   help=f"Reject bbox if it covers more than this fraction of the image "
                        f"(default: {DEFAULT_MAX_COVERAGE})")

    # --- Model ---
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"LMStudio model name (default: {DEFAULT_MODEL})")
    p.add_argument("--base-url", default=LMSTUDIO_BASE_URL,
                   help=f"LMStudio base URL (default: {LMSTUDIO_BASE_URL})")
    p.add_argument("--prompt", type=Path, metavar="FILE",
                   help="Path to a plain-text file whose contents replace the default "
                        "user prompt sent to the model")
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                   help=f"How many times to retry the model on parse failure "
                        f"(default: {DEFAULT_RETRIES})")

    # --- Execution modes ---
    p.add_argument("--dry-run", action="store_true",
                   help="Generate previews but do not call the model or write crops")
    p.add_argument("--preview-only", action="store_true",
                   help="Only regenerate preview PNGs; skip everything else")
    p.add_argument("--bbox-only", action="store_true",
                   help="Print detected bboxes to stdout; do not write any output files")
    p.add_argument("--from-log", action="store_true",
                   help="Re-apply crops using bboxes from crop_log.json; skip model call")
    p.add_argument("--force", action="store_true",
                   help="Reprocess files even if output already exists")
    p.add_argument("--workers", type=int, default=1, metavar="N",
                   help="Number of parallel workers (default: 1)")

    return p


def main() -> None:
    args = build_parser().parse_args()

    # --- Validate ---
    if args.in_place and args.output_format:
        print("ERROR: --in-place and --output-format are mutually exclusive "
              "(format conversion in-place would change the filename).", file=sys.stderr)
        sys.exit(1)
    if args.no_backup and not args.in_place:
        print("WARNING: --no-backup has no effect without --in-place.", file=sys.stderr)

    # --- Resolve directories ---
    input_dir   = args.input_dir.resolve()
    preview_dir = (args.preview_dir or input_dir / PREVIEW_DIR_NAME).resolve()
    # output_dir is only used in default mode, but we need it for exclusion logic
    output_dir  = (args.output_dir or input_dir / OUTPUT_DIR_NAME).resolve()
    log_path    = input_dir / LOG_FILE_NAME

    # --- Load user prompt ---
    if args.prompt:
        user_prompt = args.prompt.read_text().strip()
        print(f"Using custom prompt from: {args.prompt}")
    else:
        user_prompt = DEFAULT_USER_PROMPT

    # --- Collect images ---
    inputs = args.files if args.files else [str(input_dir)]
    image_paths = collect_image_paths(inputs, output_dir,
                                      exclude_suffix=args.suffix)

    if not image_paths:
        print("No image files found.")
        sys.exit(0)

    print(f"Found {len(image_paths)} image(s) to process.")

    # --- Preview-only short circuit ---
    if args.preview_only:
        for src_path in image_paths:
            preview_path = preview_dir / f"{src_path.stem}.png"
            print(f"\nGenerating preview: {src_path.name}")
            generate_preview(src_path, preview_path)
        return

    # --- Build log bbox cache for --from-log ---
    log_bbox_cache: dict[str, list[int]] = {}
    if args.from_log:
        log_bbox_cache = build_log_cache(log_path)
        if not log_bbox_cache:
            print(f"WARNING: --from-log set but no usable entries in {log_path}",
                  file=sys.stderr)
        else:
            print(f"Loaded {len(log_bbox_cache)} cached bbox(es) from log.")

    # --- LMStudio client (skipped when every file will use cached bboxes) ---
    client: OpenAI | None = None
    if not args.from_log and not args.dry_run and not args.bbox_only:
        client = OpenAI(base_url=args.base_url, api_key=LMSTUDIO_API_KEY)
    elif not args.from_log:
        # dry-run or bbox-only without from-log still need the client
        client = OpenAI(base_url=args.base_url, api_key=LMSTUDIO_API_KEY)

    log_entries = load_log(log_path)
    log_lock = threading.Lock()
    results: list[dict] = []

    def _process_one(src_path: Path) -> dict:
        output_path = compute_output_path(src_path, args, output_dir)
        cached_bbox = log_bbox_cache.get(src_path.name) if args.from_log else None

        if args.from_log and cached_bbox is None:
            print(f"\n{'='*60}")
            print(f"Processing: {src_path.name}")
            print("  WARNING: no cached bbox in log — skipping.", file=sys.stderr)
            return {"file": src_path.name, "status": "skipped",
                    "reason": "no cached bbox in log"}

        return process_image(
            src_path=src_path,
            output_path=output_path,
            preview_dir=preview_dir,
            client=client,
            model=args.model,
            user_prompt=user_prompt,
            dry_run=args.dry_run,
            force=args.force,
            padding=args.padding,
            in_place=args.in_place,
            no_backup=args.no_backup,
            output_format=args.output_format,
            min_coverage=args.min_coverage,
            max_coverage=args.max_coverage,
            retries=args.retries,
            bbox_only=args.bbox_only,
            cached_bbox=cached_bbox,
        )

    def _record(result: dict) -> None:
        with log_lock:
            results.append(result)
            log_entries.append(result)
            save_log(log_path, log_entries)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_one, p): p for p in image_paths}
            for fut in as_completed(futures):
                try:
                    _record(fut.result())
                except Exception as exc:
                    src = futures[fut]
                    print(f"  ERROR ({src.name}): {exc}", file=sys.stderr)
                    _record({"file": src.name, "status": "error", "error": str(exc)})
    else:
        for src_path in image_paths:
            try:
                _record(_process_one(src_path))
            except Exception as exc:
                print(f"  ERROR: {exc}", file=sys.stderr)
                _record({"file": src_path.name, "status": "error", "error": str(exc)})

    # --- Summary ---
    print(f"\n{'='*60}")
    counts = {s: sum(1 for r in results if r["status"] == s)
              for s in ("ok", "skipped", "error", "dry-run", "bbox-only")}
    summary = "  ".join(f"{k}={v}" for k, v in counts.items() if v)
    print(f"Done.  {summary}")
    print(f"Log:   {log_path}")


if __name__ == "__main__":
    main()
