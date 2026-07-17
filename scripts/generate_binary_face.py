#!/usr/bin/env python3
"""
Generate an animated "binary face" GIF from a portrait photo.

Converts a photo into a matrix of 0/1 characters rendered on a dark
navy/black background with bright green/emerald digits, preserving
facial detail (eyes, hair, beard, face shape, shirt/suit) through
per-cell brightness sampling. Adds a subtle vertical scan-line
animation across a handful of frames and exports an optimized,
looping GIF.

Usage:
    python scripts/generate_binary_face.py <input_image> <output_gif> [options]

Example:
    python scripts/generate_binary_face.py assets/source-profile.png assets/binary-face-dark.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


BG_COLOR = (5, 8, 16)          # near-black navy background
DIGIT_BRIGHT = (57, 255, 140)  # emerald / bright green
DIGIT_DIM = (12, 90, 55)       # dim green for low-brightness cells
SCAN_COLOR = (170, 255, 210)   # brighter scan-line highlight


def load_font(cell_size: int) -> ImageFont.FreeTypeFont:
    """Load a monospace font sized to the grid cell, falling back to default."""
    candidates = [
        "consola.ttf",
        "DejaVuSansMono.ttf",
        "Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "C:\\Windows\\Fonts\\consolab.ttf",
        "C:\\Windows\\Fonts\\consola.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, cell_size)
        except OSError:
            continue
    return ImageFont.load_default()


def isolate_portrait(image_bgr: np.ndarray) -> np.ndarray:
    """
    Crop the frame to the detected face (with margin) so the portrait
    fills the output. Falls back to a centered square crop if no face
    is detected.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    h, w = image_bgr.shape[:2]

    if len(faces) > 0:
        # Largest detected face.
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        cx, cy = x + fw / 2, y + fh / 2
        # Expand around the face to include hair, shoulders and shirt/suit.
        side = int(max(fw, fh) * 2.6)
    else:
        cx, cy = w / 2, h / 2
        side = min(w, h)

    side = min(side, w, h)
    half = side // 2

    x0 = int(max(0, min(w - side, cx - half)))
    y0 = int(max(0, min(h - side, cy - half)))

    return image_bgr[y0:y0 + side, x0:x0 + side]


def compute_brightness_grid(image_bgr: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Downsample the image to a cols x rows grid of normalized brightness values."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    small = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def render_frame(
    brightness: np.ndarray,
    digits: np.ndarray,
    cell_size: int,
    font: ImageFont.FreeTypeFont,
    scan_row: int,
    canvas_size: int,
) -> Image.Image:
    """Render one GIF frame given a brightness grid and digit matrix."""
    rows, cols = brightness.shape
    img = Image.new("RGB", (canvas_size, canvas_size), BG_COLOR)
    draw = ImageDraw.Draw(img)

    for r in range(rows):
        for c in range(cols):
            b = brightness[r, c]
            if b < 0.12:
                continue  # keep background clean where the portrait is dark/empty

            digit = "1" if digits[r, c] else "0"
            t = min(1.0, b * 1.3)
            color = tuple(int(DIGIT_DIM[i] + (DIGIT_BRIGHT[i] - DIGIT_DIM[i]) * t) for i in range(3))

            if scan_row is not None and abs(r - scan_row) <= 1:
                blend = 1.0 - abs(r - scan_row) * 0.5
                color = tuple(int(color[i] + (SCAN_COLOR[i] - color[i]) * blend) for i in range(3))

            x = c * cell_size
            y = r * cell_size
            draw.text((x, y), digit, font=font, fill=color)

    return img


def generate_binary_face(
    input_path: str,
    output_path: str,
    size: int = 640,
    cell_size: int = 10,
    frame_count: int = 12,
    frame_duration_ms: int = 90,
    seed: int = 42,
) -> None:
    """Build the animated binary-face GIF and write it to output_path."""
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    image_bgr = cv2.imread(str(src))
    if image_bgr is None:
        raise ValueError(f"Could not read image (unsupported format?): {input_path}")

    portrait = isolate_portrait(image_bgr)
    portrait = cv2.resize(portrait, (size, size), interpolation=cv2.INTER_AREA)

    cols = size // cell_size
    rows = size // cell_size
    canvas_size = cols * cell_size

    brightness = compute_brightness_grid(portrait, cols, rows)
    font = load_font(cell_size)

    rng = np.random.default_rng(seed)
    base_digits = rng.integers(0, 2, size=(rows, cols))

    frames = []
    for i in range(frame_count):
        # Flip a small fraction of digits each frame for a subtle shimmer,
        # without a full re-randomization that would cause heavy flicker.
        digits = base_digits.copy()
        flip_mask = rng.random((rows, cols)) < 0.04
        digits[flip_mask] = 1 - digits[flip_mask]
        base_digits = digits

        scan_row = int((i / frame_count) * rows)
        frame = render_frame(brightness, digits, cell_size, font, scan_row, canvas_size)
        frames.append(frame)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an animated binary-face GIF from a portrait photo.")
    parser.add_argument("input", help="Path to the source portrait image")
    parser.add_argument("output", help="Path to write the output GIF")
    parser.add_argument("--size", type=int, default=640, help="Output canvas size in pixels (default: 640)")
    parser.add_argument("--cell-size", type=int, default=10, help="Size of each digit cell in pixels (default: 10)")
    parser.add_argument("--frames", type=int, default=12, help="Number of animation frames (default: 12)")
    parser.add_argument("--duration", type=int, default=90, help="Frame duration in milliseconds (default: 90)")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        generate_binary_face(
            args.input,
            args.output,
            size=args.size,
            cell_size=args.cell_size,
            frame_count=args.frames,
            frame_duration_ms=args.duration,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Generated {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
