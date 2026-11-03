"""Render stored boxes onto card images as a contact sheet.

  uv run python scripts/preview_boxes.py OUT.png neo1-96 hgss3-38 ... [--highlight Cyndaquil]
"""
import argparse
import sqlite3

from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("cards", nargs="+")
ap.add_argument("--highlight")
args = ap.parse_args()

conn = sqlite3.connect("data/gde.sqlite")
TW = 360
tiles = []
for cid in args.cards:
    im = Image.open(f"data/images/{cid}.webp").convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    for name, x, y, w, h, s in conn.execute(
        "SELECT e.name, b.x, b.y, b.w, b.h, b.score FROM boxes b JOIN appearances a ON a.id = b.appearance_id "
        "JOIN entities e ON e.id = a.entity_id WHERE a.card_id = ?", (cid,)
    ):
        col = (255, 60, 0) if name == args.highlight else (40, 110, 255)
        d.rectangle([x * W, y * H, (x + w) * W, (y + h) * H], outline=col, width=4)
        d.text((x * W + 5, y * H + 5), f"{name} {s or 0:.2f}", fill=col)
    tiles.append(im.resize((TW, int(TW * H / W))))

cols = 3
th = max(t.height for t in tiles)
sheet = Image.new("RGB", (TW * cols, th * ((len(tiles) + cols - 1) // cols)), "white")
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * TW, (i // cols) * th))
sheet.save(args.out)
print(args.out)
