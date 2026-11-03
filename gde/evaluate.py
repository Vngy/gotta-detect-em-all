"""Open-set identification test: can we find which species are on a card WITHOUT being told?

detect.py is given each card's known entities and only has to place them. Cards not covered by the cameo
sheet (new sets, Japanese/Chinese exclusives) have no such list, so this measures the harder task: every
region is classified over all ~1,025 species.

Datasets
  en-cameo  English cards with sheet-labeled cameos (labels: main + cameo species)
  ja-151    Japanese Pokémon Card 151 (SV2a). Labels: Japanese dexId for the main Pokémon, plus cameo labels
            transferred from English 151 (sv03.5) at the same number (≤165, where numbering agrees).

Metrics (species only)
  recall@1  labeled species that are some region's top-1 prediction
  recall@5  labeled species that appear in some region's top-5
  main@1    main Pokémon is the top-1 of some region
  cameo@5   recall@5 restricted to cameo labels — the case we care about

  uv run python -m gde.evaluate en-cameo --n 150
  uv run python -m gde.evaluate ja-151
"""
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache

import httpx
import torch

from gde.db import DATA, connect
from gde.detect import (DISTRACTORS, IMAGES, clip_image_emb, clip_text_emb, fetch_image, models, padded_crop,
                        propose, reference_emb)

RESULTS = DATA / "eval"


@cache
def species_bank():
    """(dex list, text embeddings [S+K, D], reference-art embeddings [S, D]) over every species."""
    conn = connect()
    species = conn.execute("SELECT dex, name FROM entities WHERE kind='species' ORDER BY dex").fetchall()
    dexes = [s["dex"] for s in species]
    text = torch.cat([
        clip_text_emb([f"an illustration of {s['name']}, a Pokémon" for s in species[i:i + 256]])
        for i in range(0, len(species), 256)
    ] + [clip_text_emb(DISTRACTORS)])
    with ThreadPoolExecutor(16) as pool:
        refs = list(pool.map(lambda d: reference_emb(d), dexes))
    ref = torch.stack([r if r is not None else torch.zeros_like(text[0]) for r in refs])
    return dexes, text, ref


@torch.no_grad()
def identify(img, topk=5):
    """Per region: top-k species dex numbers, most likely first."""
    dexes, text, ref = species_bank()
    cands = propose(img)
    if not cands:
        return []
    *_, clip = models()
    scale = clip.logit_scale.exp()
    crop_e = clip_image_emb([padded_crop(img, b) for b, _ in cands])
    log_text = (scale * crop_e @ text.T).log_softmax(-1)               # [C, S+K]
    is_distractor = log_text[:, len(dexes):].max(-1).values
    log_text = log_text[:, : len(dexes)]
    log_ref = (scale * crop_e @ ref.T).log_softmax(-1)                  # [C, S]
    score = 0.6 * log_text + 0.4 * log_ref
    out = []
    for c, (box, owl) in enumerate(cands):
        top = score[c].topk(topk)
        out.append({
            "box": box, "owl": owl,
            "top": [dexes[i] for i in top.indices.tolist()],
            "p": top.values.exp().tolist(),
            "background": bool(is_distractor[c] > log_text[c].max()),
        })
    return out


def en_cameo(n, seed):
    conn = connect()
    cards = conn.execute(
        "SELECT DISTINCT c.id, c.image FROM cards c JOIN appearances a ON a.card_id = c.id "
        "WHERE a.role = 'cameo' AND c.image IS NOT NULL ORDER BY c.id"
    ).fetchall()
    random.Random(seed).shuffle(cards)
    items = []
    for c in cards[:n]:
        labels = conn.execute(
            "SELECT e.dex, a.role FROM appearances a JOIN entities e ON e.id = a.entity_id "
            "WHERE a.card_id = ? AND e.kind = 'species'", (c["id"],)
        ).fetchall()
        if labels:
            items.append({"id": c["id"], "url": f"{c['image']}/high.webp", "path": IMAGES / f"{c['id']}.webp",
                          "labels": {l["dex"]: l["role"] for l in labels}})
    return items


def ja_151():
    conn = connect()
    ja = httpx.get("https://api.tcgdex.net/v2/ja/sets/SV2a", timeout=60).json()
    cache_dir = DATA / "tcgdex-ja" / "cards"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def card(cid):
        f = cache_dir / f"{cid}.json"
        if not f.exists():
            f.write_text(httpx.get(f"https://api.tcgdex.net/v2/ja/cards/{cid}", timeout=60).text)
        return json.loads(f.read_text())

    with ThreadPoolExecutor(16) as pool:
        cards = list(pool.map(card, [c["id"] for c in ja["cards"]]))
    items = []
    for c in cards:
        if not c.get("image"):
            continue
        labels = {d: "main" for d in c.get("dexId") or []}
        num = int(c["localId"])
        if num <= 165:
            for row in conn.execute(
                "SELECT e.dex FROM appearances a JOIN entities e ON e.id = a.entity_id "
                "JOIN cards en ON en.id = a.card_id WHERE en.set_id = 'sv03.5' AND CAST(en.local_id AS INTEGER) = ? "
                "AND a.role = 'cameo' AND e.kind = 'species'", (num,)
            ):
                labels.setdefault(row["dex"], "cameo")
        if labels:
            items.append({"id": f"ja-{c['id']}", "url": f"{c['image']}/high.webp",
                          "path": DATA / "images-ja" / f"{c['id']}.webp", "labels": labels})
    return items


def run(name, items):
    with ThreadPoolExecutor(8) as pool:
        imgs = pool.map(lambda it: fetch_image(it["url"], it["path"]), items)
        totals = {k: [0, 0] for k in ("recall@1", "recall@5", "main@1", "cameo@5")}
        rows = []
        t0 = time.time()
        for it, img in zip(items, imgs):
            if img is None:
                continue
            regions = identify(img)
            top1 = {r["top"][0] for r in regions}
            top5 = {d for r in regions for d in r["top"]}
            for dex, role in it["labels"].items():
                checks = {"recall@1": dex in top1, "recall@5": dex in top5}
                checks["main@1" if role == "main" else "cameo@5"] = dex in (top1 if role == "main" else top5)
                for k, hit in checks.items():
                    totals[k][0] += hit
                    totals[k][1] += 1
            rows.append({"id": it["id"], "labels": it["labels"],
                         "regions": [{k: v for k, v in r.items() if k != "box"} | {"box": list(r["box"])}
                                     for r in regions]})
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(rows, indent=1))
    print(f"\n{name}: {len(rows)} cards in {time.time() - t0:.0f}s")
    for k, (hit, n) in totals.items():
        if n:
            print(f"  {k:9s} {hit / n:6.1%}  ({hit}/{n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["en-cameo", "ja-151"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    items = en_cameo(args.n, args.seed) if args.dataset == "en-cameo" else ja_151()
    print(f"{args.dataset}: {len(items)} labeled cards")
    run(args.dataset, items)


if __name__ == "__main__":
    main()
