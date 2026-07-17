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
DIGIT_BRIGHT = (120, 255, 170) # emerald / bright green highlights
DIGIT_DIM = (10, 60, 38)       # dim green for low-brightness cells (still readable)
SCAN_COLOR = (200, 255, 225)   # brighter scan-line highlight


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


def segment_foreground(image_bgr: np.ndarray) -> np.ndarray:
    """
    Build a soft head-and-shoulders silhouette mask so the backdrop
    renders as clean dark navy while the subject (hair, face, shirt,
    suit) fills the frame.

    Studio headshot backdrops are frequently lit with a gradient/vignette
    rather than a flat color, which makes pixel-color background removal
    (chroma-key style, or GrabCut with a generic rectangle) unreliable.
    Instead we re-run face detection on the already-cropped portrait and
    draw a deterministic bust shape (an ellipse for the head, a widening
    trapezoid for the shoulders) sized off the detected face box, then
    feather the edges. This is robust across lighting/backdrop styles and
    always produces a clean, recognizable silhouette.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) > 0:
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        fcx, fcy = fx + fw / 2, fy + fh / 2
    else:
        fw, fh = w * 0.38, h * 0.38
        fcx, fcy = w / 2, h * 0.42

    mask = np.zeros((h, w), np.float32)

    # Head: ellipse a little taller than the raw face box to include hair.
    head_rx, head_ry = int(fw * 0.85), int(fh * 1.15)
    head_cy = int(fcy - fh * 0.05)
    cv2.ellipse(mask, (int(fcx), head_cy), (head_rx, head_ry), 0, 0, 360, 1.0, -1)

    # Shoulders: trapezoid widening from just below the chin to the bottom
    # of the frame, wide enough to read as a shirt/suit silhouette.
    chin_y = int(fcy + fh * 0.65)
    shoulder_half_top = fw * 0.62
    shoulder_half_bottom = fw * 1.55
    poly = np.array(
        [
            [fcx - shoulder_half_top, chin_y],
            [fcx + shoulder_half_top, chin_y],
            [min(w, fcx + shoulder_half_bottom), h],
            [max(0, fcx - shoulder_half_bottom), h],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, poly, 1.0)

    feather = max(3, int(min(h, w) * 0.015)) | 1  # odd kernel size
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    return np.clip(mask, 0.0, 1.0)


def compute_brightness_grid(image_bgr: np.ndarray, cols: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample the image to a cols x rows grid of normalized brightness
    values, restricted to the segmented foreground (the person), so the
    backdrop renders as clean empty space rather than digit noise.

    Returns (brightness, mask) where mask marks which cells belong to the
    subject.
    """
    fg_mask = segment_foreground(image_bgr)

    gray_u8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32)

    # Contrast-stretch using only foreground pixel statistics so the
    # backdrop (often mid-gray) doesn't skew the range used by the face,
    # hair, shirt and suit. This preserves the overall shading (dark suit,
    # mid-tone skin, bright shirt).
    fg_pixels = gray[fg_mask.astype(bool)]
    if fg_pixels.size == 0:
        fg_pixels = gray.reshape(-1)
    lo, hi = np.percentile(fg_pixels, 2), np.percentile(fg_pixels, 98)
    if hi - lo < 1e-3:
        hi = lo + 1e-3
    stretched = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    stretched = np.power(stretched, 0.85)

    # Blend in CLAHE local contrast so fine features (eyes, brows, beard,
    # hairline) stay distinguishable instead of collapsing into flat skin
    # tone once downsampled to a coarse digit grid.
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(10, 10))
    local_detail = clahe.apply(gray_u8).astype(np.float32) / 255.0

    combined = np.clip(0.5 * stretched + 0.5 * local_detail, 0.0, 1.0)

    # Punch up overall contrast around the midpoint so eyes, brows, beard
    # and shirt/suit edges read clearly instead of blending into a flat
    # mid-tone once reduced to a coarse digit grid.
    combined = np.clip((combined - 0.5) * 1.6 + 0.5, 0.0, 1.0)

    small = cv2.resize(combined, (cols, rows), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(fg_mask.astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA)

    return small, small_mask


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
            if b < 0.05:
                continue  # keep background clean where the portrait is truly empty

            digit = "1" if digits[r, c] else "0"
            t = min(1.0, b) ** 0.7
            t = 0.15 + 0.85 * t  # floor so mid/low tones stay clearly readable
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
    cell_size: int = 8,
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

    brightness, fg_mask = compute_brightness_grid(portrait, cols, rows)
    # Zero out the backdrop so only the person (hair, face, shirt, suit)
    # renders as digits, keeping the background clean dark navy.
    brightness = brightness * np.clip(fg_mask * 1.6, 0.0, 1.0)
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
    parser.add_argument("--cell-size", type=int, default=8, help="Size of each digit cell in pixels (default: 8)")
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
