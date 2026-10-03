#!/usr/bin/env python3
"""Minimal self-check: run `uv run test_crop.py`."""
from pathlib import Path
from PIL import Image
from types import SimpleNamespace
import crop


def _fake_client(content, finish_reason, tokens):
    resp = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content),
            finish_reason=finish_reason)],
        usage=SimpleNamespace(completion_tokens=tokens))
    return SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kw: resp)))


def test_call_model(tmp_png):
    ok = crop.call_model(_fake_client('{"x1":1,"y1":2,"x2":3,"y2":4}\n', "stop", 20),
                         "m", tmp_png, "p")
    assert ok == '{"x1":1,"y1":2,"x2":3,"y2":4}', ok

    # Garbage from a broken backend must not look like a parse failure.
    try:
        crop.call_model(_fake_client("?" * 256, "length", 256), "m", tmp_png, "p")
    except ValueError as exc:
        assert "finish_reason='length'" in str(exc), exc
    else:
        raise AssertionError("truncated completion should raise")


def test_parse():
    assert crop.parse_bbox_from_response('{"x1": 1, "y1": 2, "x2": 3, "y2": 4}') == (1, 2, 3, 4)
    assert crop.parse_bbox_from_response('<|box_start|>(1,2),(3,4)<|box_end|>') == (1, 2, 3, 4)
    for bad in ('{"x1": 215, "y1": 0', "?" * 64):
        try:
            crop.parse_bbox_from_response(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should have raised: {bad!r}")


def test_collect_and_mirror(root):
    """Recursion must mirror structure, and must not eat its own output."""
    for rel in ("a.tif", "sub/a.tif", "sub/deep/b.png",
                "output/a.tif", "previews/sub/a.png", "c.orig.tif"):
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4)).save(f)

    excl = (root / "output", root / "previews")

    flat = crop.collect_image_paths([str(root)], exclude_dirs=excl)
    assert [f.name for f in flat] == ["a.tif"], flat

    deep = crop.collect_image_paths([str(root)], exclude_dirs=excl, recursive=True)
    keys = sorted(str(crop.relative_key(f, root.resolve())) for f in deep)
    # output/ and previews/ excluded; the .orig backup skipped.
    assert keys == ["a.tif", "sub/a.tif", "sub/deep/b.png"], keys

    # A ** glob recurses without the flag and gets the same exclusions.
    globbed = crop.collect_image_paths([f"{root}/**/*.tif"], exclude_dirs=excl)
    assert sorted(str(crop.relative_key(f, root.resolve())) for f in globbed) \
        == ["a.tif", "sub/a.tif"], globbed

    # Same basename in two directories must not collide in the output tree.
    args = SimpleNamespace(in_place=False, suffix=None, output_format=None)
    out = root / "out"
    a = crop.compute_output_path(root / "a.tif", args, out, Path("a.tif"))
    b = crop.compute_output_path(root / "sub/a.tif", args, out, Path("sub/a.tif"))
    assert a == out / "a.tif", a
    assert b == out / "sub" / "a.tif", b
    assert a != b

    # Format override still applies through the mirrored path.
    args.output_format = "jpg"
    assert crop.compute_output_path(root / "sub/a.tif", args, out,
                                    Path("sub/a.tif")) == out / "sub" / "a.jpg"

    # Files outside --input-dir fall back to flat, as before.
    assert crop.relative_key(Path("/elsewhere/x.tif"), root.resolve()) == Path("x.tif")


if __name__ == "__main__":
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "x.png"
        Image.new("RGB", (4, 4)).save(p)
        test_call_model(p)
    test_parse()
    with tempfile.TemporaryDirectory() as d:
        test_collect_and_mirror(pathlib.Path(d))
    print("ok")
