"""Suggest appearances the source lists don't have, using open-set identification (evaluate.identify + probe).

A region becomes a suggestion when its top-1 species has p >= MIN_P (≈80–85% precision lower bound on en-cameo
at 0.3), it isn't classified as background, it doesn't overlap a box already placed on the card, and that
species isn't already linked to the card. Suggestions are reviewed in the UI; accepting one creates a cameo
appearance with a verified box. Rejected suggestions are kept so they are never re-suggested.

  uv run python -m gde.discover --limit 500
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from gde import evaluate
from gde.db import connect
from gde.detect import card_image, iou

MIN_P = 0.3
MODEL = "owlv2+clip+probe-v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--min-p", type=float, default=MIN_P)
    ap.add_argument("--entity", help="only cards this entity is already linked to")
    args = ap.parse_args()
    evaluate.USE_PROBE = True
    conn = connect()
    sql = ("SELECT c.id, c.image FROM cards c WHERE c.image IS NOT NULL "
           "AND c.id NOT IN (SELECT card_id FROM discover_scans WHERE model = ?)")
    params = [MODEL]
    if args.entity:
        sql += (" AND c.id IN (SELECT a.card_id FROM appearances a JOIN entities e ON e.id = a.entity_id "
                "WHERE e.name = ? COLLATE NOCASE)")
        params.append(args.entity)
    # Cameos cluster on Trainer cards and illustration-style rares, so scan those first.
    order = (" ORDER BY c.category = 'Trainer' DESC, (c.rarity LIKE '%Illustration%' OR c.rarity LIKE '%Art%') DESC,"
             " c.id")
    cards = conn.execute(sql + order, params).fetchall()[: args.limit]
    print(f"{len(cards)} cards to scan", flush=True)
    pool = ThreadPoolExecutor(8)
    images = pool.map(lambda c: card_image(c["id"], c["image"]), cards)
    t0, found = time.time(), 0
    for i, (card, img) in enumerate(zip(cards, images), 1):
        if img is None:
            continue  # not marked scanned: retried once the image exists
        W, H = img.size
        linked = {r[0] for r in conn.execute("SELECT entity_id FROM appearances WHERE card_id = ?", (card["id"],))}
        placed = [((x * W, y * H, (x + w) * W, (y + h) * H))
                  for x, y, w, h in conn.execute(
                      "SELECT b.x, b.y, b.w, b.h FROM boxes b JOIN appearances a ON a.id = b.appearance_id "
                      "WHERE a.card_id = ?", (card["id"],))]
        new = []
        for r in evaluate.identify(img):
            ent = f"species:{r['top'][0]}"
            if r["background"] or r["p"][0] < args.min_p or ent in linked:
                continue
            if any(iou(r["box"], b) > 0.3 for b in placed):
                continue
            x0, y0, x1, y1 = r["box"]
            new.append((card["id"], ent, x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H, r["p"][0], MODEL))
            linked.add(ent)
        with conn:
            conn.executemany(
                "INSERT OR IGNORE INTO suggestions (card_id, entity_id, x, y, w, h, p, model) VALUES (?,?,?,?,?,?,?,?)",
                new,
            )
            conn.execute("INSERT OR REPLACE INTO discover_scans VALUES (?, ?)", (card["id"], MODEL))
        found += len(new)
        if i % 100 == 0 or i == len(cards):
            rate = i / (time.time() - t0)
            print(f"[{i}/{len(cards)}] {found} suggestions, {rate:.2f} cards/s, "
                  f"eta {(len(cards) - i) / rate / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
