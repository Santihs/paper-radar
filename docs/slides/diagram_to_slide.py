"""Render an archify diagram as a slide image in the deck palette (black + OpenRouter lime).

The validated archify HTML is never modified: a recoloured copy is written to a temp dir,
captured with headless Chrome and auto-cropped to the drawing.

    uv run --no-project --with pillow python docs/slides/diagram_to_slide.py IN.html OUT.png
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BG = "#07080A"

# Deck tokens: bg #07080A, card #111317, line #262A31, text #F4F4F5, soft #A1A7B3,
# lime #C5F82A (main path, OpenRouter), red #F87171 (policy, rejections).
PALETTE = f"""
html, html[data-theme], html[data-preset][data-theme], :root {{
  --bg: {BG} !important; --grid: #16191e !important;
  --canvas-dot: rgba(255,255,255,.035) !important;
  --text: #F4F4F5 !important; --text-muted: #A1A7B3 !important; --text-dim: #5C6270 !important;
  --text-faint: #7A808C !important; --panel: #0B0D10 !important; --panel-border: #262A31 !important;
  --lane-fill: rgba(17,19,23,.55) !important; --lane-stroke: #262A31 !important;
  --arrow: #5C6270 !important; --arrow-emphasis: #C5F82A !important; --mask: #0B0D10 !important;
  --frontend-fill: #111317 !important;   --frontend-stroke: #F4F4F5 !important;
  --backend-fill: rgba(197,248,42,.10) !important; --backend-stroke: #C5F82A !important;
  --security-fill: rgba(248,113,113,.10) !important; --security-stroke: #F87171 !important;
  --database-fill: #111317 !important;   --database-stroke: #A1A7B3 !important;
  --cloud-fill: #111317 !important;      --cloud-stroke: #D4D4D8 !important;
  --messagebus-fill: rgba(197,248,42,.06) !important; --messagebus-stroke: #9CCB22 !important;
  --external-fill: #111317 !important;   --external-stroke: #5C6270 !important;
}}
body {{ background: {BG} !important; }}
.toolbar, .cards, .no-print, .header, g[data-legend] {{ display: none !important; }}
.diagram-container {{ border: none !important; background: transparent !important; }}
"""


def render(src: Path, out: Path, width: int = 2048, height: int = 1320) -> None:
    html = src.read_text(encoding="utf-8")
    html = html.replace("<html", '<html data-theme="dark"', 1)
    html = html.replace("</head>", f"<style>{PALETTE}</style></head>", 1)
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "slide.html"
        page.write_text(html, encoding="utf-8")
        shot = Path(tmp) / "shot.png"
        subprocess.run(
            [
                CHROME,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=1",
                f"--window-size={width},{height}",
                "--virtual-time-budget=6000",
                f"--screenshot={shot}",
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
        )
        img = Image.open(shot).convert("RGB")
    bbox = (
        ImageChops.difference(img, Image.new("RGB", img.size, BG))
        .point(lambda v: 255 if v > 18 else 0)
        .getbbox()
    )
    pad = 12
    if bbox:
        left, top, right, bottom = bbox
        img = img.crop(
            (
                max(left - pad, 0),
                max(top - pad, 0),
                min(right + pad, img.width),
                min(bottom + pad, img.height),
            )
        )
    img.save(out)
    print(out, img.size)


if __name__ == "__main__":
    render(Path(sys.argv[1]), Path(sys.argv[2]))
