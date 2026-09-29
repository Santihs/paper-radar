"""Recolor slide images into the deck palette (black + OpenRouter lime).

ui    : light UI screenshots -> dark mode. Luminance is inverted (light bg -> black,
        dark text -> white); saturated accents are remapped by hue to deck roles
        (primary -> lime, secondary -> grey, warning -> red).
illus : flat illustrations -> grey figures on black, lime only where the message is.
shift : (--shift) transparent illustrations keep their art; only chosen hues rotate to lime.
        This is the one used in the deck: the others degrade low-resolution art.

    uv run --no-project --with pillow --with numpy python docs/slides/recolor.py [--shift]
"""

from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parents[2] / "img"
OUT = Path(__file__).resolve().parent / "img"

BG = np.array([0x0B, 0x0D, 0x10], float)
CARD = np.array([0x16, 0x19, 0x1E], float)
TEXT = np.array([0xF4, 0xF4, 0xF5], float)
LIME = np.array([0xC5, 0xF8, 0x2A], float)
SOFT = np.array([0xA1, 0xA7, 0xB3], float)
RED = np.array([0xF8, 0x71, 0x71], float)


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(SRC / name)
    if img.mode in (
        "RGBA",
        "LA",
        "P",
    ):  # transparent bg -> white, maps to deck bg
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, "white")
        img = Image.alpha_composite(white, img)
    img = img.convert("RGB")
    rgb = np.asarray(img, float) / 255
    hsv = np.asarray(img.convert("HSV"), float) / 255
    return rgb, hsv


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def hue_role(h: np.ndarray, roles: dict[str, str]) -> np.ndarray:
    """Map hue (0..1) to an accent colour per pixel using named hue bands."""
    deg = h * 360
    bands = {
        "orange": (deg < 45) | (deg >= 345),
        "green": (deg >= 90) & (deg < 200),
        "purple": (deg >= 200) & (deg < 345),
    }
    colours = {"lime": LIME, "soft": SOFT, "red": RED}
    out = np.zeros((*h.shape, 3))
    for band, role in roles.items():
        out[bands[band]] = colours[role]
    return out


def ui(name: str, out: str, roles: dict[str, str]) -> None:
    rgb, hsv = load(name)
    lum = luminance(rgb)[..., None]
    base = TEXT + (BG - TEXT) * lum  # inverted: white -> BG, black -> TEXT
    s, v = hsv[..., 1:2], hsv[..., 2:3]
    weight = np.clip((s - 0.18) / 0.35, 0, 1) * np.clip((v - 0.25) / 0.4, 0, 1)
    accent = hue_role(hsv[..., 0], roles)
    # keep a little of the original lightness so gradients survive
    accent = accent * (0.75 + 0.25 * v)
    result = base * (1 - weight) + accent * weight
    save(result, out)


def background(lum: np.ndarray, sat: np.ndarray, threshold: float) -> np.ndarray:
    """Near-white pixels connected to the image border (so white objects inside survive)."""
    near_white = (lum > threshold) & (sat < 0.15)
    bg = np.zeros_like(near_white)
    bg[[0, -1], :] = near_white[[0, -1], :]  # seed from the whole border
    bg[:, [0, -1]] = near_white[:, [0, -1]]
    while True:  # grow into 4-connected near-white neighbours until stable
        grown = bg.copy()
        grown[1:, :] |= bg[:-1, :]
        grown[:-1, :] |= bg[1:, :]
        grown[:, 1:] |= bg[:, :-1]
        grown[:, :-1] |= bg[:, 1:]
        grown &= near_white
        if (grown == bg).all():
            return bg
        bg = grown


def illus(name: str, out: str, accent: str, bg_threshold: float, fill_holes: bool) -> None:
    """Neutral grey figures on the deck background; lime only where the message is.

    accent="top": everything saturated in the upper third (speech bubbles).
    accent="warm": red/orange elements (stress cues: shirt, bolts, hourglass).
    """
    rgb, hsv = load(name)
    lum = luminance(rgb)
    hue, sat = hsv[..., 0] * 360, hsv[..., 1]

    stops = np.array(
        [[0x2A, 0x30, 0x38], [0x5C, 0x62, 0x70], [0xA1, 0xA7, 0xB3], [0xE4, 0xE4, 0xE7]], float
    )
    t = np.clip(lum, 0, 1) * (len(stops) - 1)
    i = np.clip(t.astype(int), 0, len(stops) - 2)
    f = (t - i)[..., None]
    result = stops[i] * (1 - f) + stops[i + 1] * f

    bg = background(lum, sat, bg_threshold)
    if fill_holes:  # white gaps enclosed by chairs or pots are background too, not objects
        bg |= (lum > 0.86) & (sat < 0.15)
    print(f"  {name}: background {bg.mean():.0%} (corner lum {lum[0, 0]:.2f}, sat {sat[0, 0]:.2f})")
    panel = ~bg & (lum > 0.78) & (lum <= 0.93) & (sat < 0.25)
    result[panel] = CARD

    saturated = (sat > 0.30) & (lum > 0.25) & ~bg & ~panel
    if accent == "top":
        h, w = lum.shape
        rows = np.arange(h)[:, None] < h * 0.31  # bubbles sit above the heads
        cols = np.arange(w)[None, :] > w * 0.26  # skip the plant leaves on the left
        mask = saturated & rows & cols & (lum > 0.35)
    else:
        mask = saturated & ((hue < 40) | (hue >= 340))
    weight = np.clip((sat - 0.30) / 0.25, 0, 1)[..., None] * mask[..., None]
    lime = LIME * (0.7 + 0.3 * np.clip(lum, 0, 1))[..., None]
    result = result * (1 - weight) + lime * weight
    result[bg] = BG
    save(result, out)


LIME_HUE = 75 / 360  # hue of #C5F82A


def shift(name: str, out: str, to_lime: tuple[float, float], red: str | None) -> None:
    """Rotate chosen hues to the deck lime, keeping saturation, shading and transparency intact.

    to_lime: hue band in degrees (start, end) moved to lime (e.g. blues/purples/teals).
    red: "grey" desaturates strong reds (the shirt); "lime" turns them lime (a speech bubble).
    Skin is less saturated than these reds, so it is left alone either way.
    """
    img = Image.open(SRC / name).convert("RGBA")
    alpha = img.getchannel("A")
    hsv = np.asarray(img.convert("RGB").convert("HSV"), float)
    h, s, v = hsv[..., 0] / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    deg = h * 360
    in_band = (deg >= to_lime[0]) & (deg < to_lime[1])
    band = in_band & (s > 0.12)
    h = np.where(band, LIME_HUE, h)
    s = np.where(band, np.clip(s * 1.3, 0, 0.9), s)  # closer to the deck lime, less olive
    s = np.where(in_band & ~band, 0.0, s)  # pale lavender tints (background blob) -> neutral grey
    reds = ((deg < 16) | (deg >= 345)) & (s > 0.42)
    if red == "grey":
        s = np.where(reds, 0.06, s)
        v = np.where(reds, v * 0.62, v)
    elif red == "lime":  # bubbles only: cheeks and lips keep their colour
        above = np.arange(h.shape[0])[:, None] < h.shape[0] * 0.31
        bubble = ((deg < 25) | (deg >= 340)) & (s > 0.18) & above
        h = np.where(bubble, LIME_HUE, h)
        s = np.where(bubble, np.clip(s * 1.3, 0, 0.9), s)
    merged = np.stack([h, s, v], axis=-1) * 255
    rgb = Image.fromarray(merged.astype("uint8"), "HSV").convert("RGB")
    rgb.putalpha(alpha)
    OUT.mkdir(exist_ok=True)
    rgb.save(OUT / out)
    print(OUT / out)


def save(arr: np.ndarray, out: str) -> None:
    OUT.mkdir(exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).save(OUT / out)
    print(OUT / out)


if __name__ == "__main__" and "--shift" in __import__("sys").argv:
    # original illustrations, recoloured by hue rotation only (transparent PNGs)
    shift(
        "overwhelmed-engineer.png", "overwhelmed-engineer-green.png", to_lime=(140, 300), red="grey"
    )
    shift("team discussion.png", "team-discussion-green.png", to_lime=(140, 300), red="lime")
elif __name__ == "__main__":
    ui(
        "routing-two-layers.png",
        "routing-two-layers.png",
        {"purple": "lime", "green": "soft", "orange": "red"},
    )
    ui(
        "routing-provider-weghting.png",
        "routing-provider-weighting.png",
        {"green": "lime", "purple": "soft", "orange": "red"},
    )
    ui(
        "routing-model-fallback.png",
        "routing-model-fallback.png",
        {"purple": "lime", "green": "soft", "orange": "red"},
    )
    illus(
        "team discussion.png",
        "team-discussion.png",
        accent="top",
        bg_threshold=0.88,
        fill_holes=True,
    )
    # higher threshold: the pale blob behind him stays a panel, so the white papers on it survive
    illus(
        "overwhelmed-engineer.png",
        "overwhelmed-engineer.png",
        accent="warm",
        bg_threshold=0.95,
        fill_holes=False,
    )
