"""Regenerate the Moth logo assets from the source artwork.

The source is a black moth on a white square inside a black frame, 1254x1254, no alpha. That shape
is only usable as-is on a light background, so this produces the variants the repo and the model
card need, and nothing else:

    logo-tile.png    512x512  white tile, black moth — works on light *and* dark themes, so it is
                              what the model card uses (Hugging Face markdown cannot switch by theme)
    logo.png         512x512  transparent, black moth — light backgrounds, favicon source
    logo-white.png   512x512  transparent, white moth — dark backgrounds
    favicon.png       64x64   transparent, black moth — the demo page's tab icon

    uv run --with pillow --with numpy python docs/assets/make_logo.py ~/Downloads/moth.png

The alpha channel is `255 - luminance`, which is exact for flat artwork: the white background
becomes fully transparent, the black moth fully opaque, and the anti-aliased edges keep their
coverage. Nothing is resampled before the crop, so no halo is introduced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
SIZES = {"logo-tile.png": 512, "logo.png": 512, "logo-white.png": 512, "favicon.png": 64}
#: Fraction of the final square left empty around the moth, so the mark never touches the edge.
PADDING = 0.08


def crop_to_artwork(source: Image.Image) -> Image.Image:
    """Drop the black frame, keep the white square."""
    grey = np.asarray(source.convert("L"))
    white = grey > 200
    ys, xs = np.where(white)
    if not len(xs):
        raise SystemExit("the source has no light area — is it the right file?")
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    print(f"  white square: {box[2] - box[0]}x{box[3] - box[1]} at {box[0]},{box[1]}")
    return source.convert("L").crop(box)


def to_rgba(mark: Image.Image) -> Image.Image:
    """Luminance to alpha: white becomes transparent, black stays opaque."""
    grey = np.asarray(mark)
    alpha = 255 - grey
    rgba = np.zeros((*grey.shape, 4), dtype=np.uint8)
    rgba[..., 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")


def square_with_padding(mark: Image.Image) -> Image.Image:
    """Trim to the mark, then pad to a square with PADDING around it."""
    bbox = mark.getbbox()
    if bbox is None:
        raise SystemExit("empty artwork after alpha extraction")
    trimmed = mark.crop(bbox)
    side = int(max(trimmed.size) * (1 + 2 * PADDING))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(trimmed, ((side - trimmed.width) // 2, (side - trimmed.height) // 2), trimmed)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(prog="make_logo.py")
    parser.add_argument("source", help="the source PNG (black moth on white, framed)")
    args = parser.parse_args()

    source = Image.open(args.source)
    print(f"source: {source.size[0]}x{source.size[1]} {source.mode}")
    mark = square_with_padding(to_rgba(crop_to_artwork(source)))
    print(f"mark: {mark.size[0]}x{mark.size[1]} square, alpha kept")

    tile = Image.new("RGB", mark.size, (255, 255, 255))
    tile.paste(mark, (0, 0), mark)

    for name, side in SIZES.items():
        target = mark.resize((side, side), Image.LANCZOS)
        if name == "logo-tile.png":
            resized = tile.resize((side, side), Image.LANCZOS)
        elif name == "logo-white.png":
            white = np.array(target).copy()
            white[..., :3] = 255  # keep alpha, paint the mark white
            resized = Image.fromarray(white, mode="RGBA")
        else:
            resized = target
        path = OUT / name
        resized.save(path, optimize=True)
        print(f"  {name:16s} {side}x{side}  {path.stat().st_size / 1024:6.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())