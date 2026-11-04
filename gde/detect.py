"""Locate known entities in card art.

We already know *which* entities are on a card (TCGdex main Pokémon + cameo sheet), so this is an
assignment problem, not open classification:
  1. OWLv2 proposes candidate regions with generic prompts (creature / person).
  2. CLIP scores every (entity, region) pair: text similarity to the entity's name, plus, for species,
     image similarity to its official artwork.
  3. A one-to-one assignment picks a box per entity. Manual/verified boxes are never overwritten.

Usage:
  uv run python -m gde.detect --entity cyndaquil
  uv run python -m gde.detect --card neo1-96
"""
import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache

import httpx
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import CLIPModel, CLIPProcessor, Owlv2ForObjectDetection, Owlv2Processor

from gde.db import DATA, connect

IMAGES = DATA / "images"
REFS = DATA / "refs"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OWL_ID = "google/owlv2-base-patch16-ensemble"
CLIP_ID = "openai/clip-vit-large-patch14"
PROMPTS = ["a pokemon", "a cartoon creature", "an animal", "a person", "a human character", "a monster"]
OFFICIAL_ART = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{dex}.png"


@cache
def models():
    owl_p = Owlv2Processor.from_pretrained(OWL_ID)
    owl = Owlv2ForObjectDetection.from_pretrained(OWL_ID).to(DEVICE).eval()
    clip_p = CLIPProcessor.from_pretrained(CLIP_ID)
    clip = CLIPModel.from_pretrained(CLIP_ID).to(DEVICE).eval()
    return owl_p, owl, clip_p, clip


def fetch_image(url: str, path) -> Image.Image | None:
    if not path.exists():
        r = httpx.get(url, timeout=30, follow_redirects=True)
        if r.status_code != 200:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    return Image.open(io.BytesIO(path.read_bytes())).convert("RGB")


def card_image(card_id, image_base):
    return fetch_image(f"{image_base}/high.webp", IMAGES / f"{card_id}.webp") if image_base else None


def prefetched(fn, items, ahead=16, workers=8):
    """Yield (item, fn(item)) in order, running fn at most `ahead` items early.

    Bounded on purpose: submitting everything up front kept every decoded card image alive in its future
    (~1.5 MB each) and grew a full --all run to 37 GB.
    """
    from collections import deque
    with ThreadPoolExecutor(workers) as pool:
        it, window = iter(items), deque()
        for item in it:
            window.append((item, pool.submit(fn, item)))
            if len(window) >= ahead:
                break
        while window:
            item, fut = window.popleft()
            nxt = next(it, None)
            if nxt is not None:
                window.append((nxt, pool.submit(fn, nxt)))
            yield item, fut.result()


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0


@torch.no_grad()
def propose(img: Image.Image, max_boxes=20, threshold=0.04):
    owl_p, owl, *_ = models()
    inputs = owl_p(text=[PROMPTS], images=img, return_tensors="pt").to(DEVICE)
    out = owl(**inputs)
    # OWLv2 pads to square; target size is the padded square side.
    side = max(img.size)
    res = owl_p.post_process_grounded_object_detection(out, threshold=threshold, target_sizes=[(side, side)])[0]
    W, H = img.size
    cands = []
    for score, box in sorted(zip(res["scores"].tolist(), res["boxes"].tolist()), reverse=True):
        x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
        area = (x1 - x0) * (y1 - y0) / (W * H)
        if area < 0.002 or area > 0.85:
            continue
        if y1 < 0.10 * H:  # header strip: name bar, HP, evolution icon
            continue
        if y0 > 0.85 * H:  # footer strip: weakness/retreat bar, set info, copyright
            continue
        if (x1 - x0) > 4 * (y1 - y0) or (y1 - y0) > 4 * (x1 - x0):  # text bars and borders, not creatures
            continue
        if any(iou((x0, y0, x1, y1), c[0]) > 0.6 for c in cands):
            continue
        cands.append(((x0, y0, x1, y1), score))
        if len(cands) >= max_boxes:
            break
    return cands


def _features(out):
    # transformers >=5 returns a ModelOutput whose pooler_output is the projected embedding.
    return out if isinstance(out, torch.Tensor) else out.pooler_output


@torch.no_grad()
def clip_image_emb(images):
    *_, clip_p, clip = models()
    e = _features(clip.get_image_features(**clip_p(images=images, return_tensors="pt").to(DEVICE)))
    return torch.nn.functional.normalize(e, dim=-1)


@torch.no_grad()
def clip_text_emb(texts):
    *_, clip_p, clip = models()
    e = _features(clip.get_text_features(**clip_p(text=texts, return_tensors="pt", padding=True).to(DEVICE)))
    return torch.nn.functional.normalize(e, dim=-1)


@cache
def reference_emb(dex: int):
    path = REFS / f"{dex}.png"
    if fetch_image(OFFICIAL_ART.format(dex=dex), path) is None:
        return None
    rgba = Image.open(path).convert("RGBA")
    flat = Image.new("RGB", rgba.size, "white")
    flat.paste(rgba, mask=rgba.split()[3])
    return clip_image_emb([flat])[0]


DISTRACTORS = ["the frame or border of a trading card", "an icon or symbol on a trading card", "printed text",
               "background scenery", "a tree or plants", "an object or item"]


def entity_prompt(ent):
    if ent["kind"] == "species":
        return f"an illustration of {ent['name']}, a Pokémon"
    return f"an illustration of {ent['name']}, a Pokémon trainer character"


def padded_crop(img, box, pad=0.15):
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return img.crop((int(max(0, x0 - px)), int(max(0, y0 - py)),
                     int(min(img.width, x1 + px)), int(min(img.height, y1 + py))))


@torch.no_grad()
def score_matrix(img, cands, appearances):
    """log P(entity | crop), [E, C]: zero-shot over the card's expected entities plus distractor labels,
    blended for species with similarity to official artwork."""
    *_, clip = models()
    crop_e = clip_image_emb([padded_crop(img, b) for b, _ in cands])               # [C, D]
    text_e = clip_text_emb([entity_prompt(a) for a in appearances] + DISTRACTORS)   # [E+K, D]
    scale = clip.logit_scale.exp()
    log_p_text = (scale * crop_e @ text_e.T).log_softmax(dim=-1).T[: len(appearances)]  # [E, C]
    refs = [reference_emb(a["dex"]) if a["kind"] == "species" else None for a in appearances]
    if any(r is not None for r in refs):
        ref_e = torch.stack([r if r is not None else torch.zeros_like(crop_e[0]) for r in refs])
        ref_sim = scale * ref_e @ crop_e.T                                           # [E, C]
        log_p_ref = ref_sim.log_softmax(dim=-1)  # which crop looks most like this entity's artwork
        for i, r in enumerate(refs):
            if r is not None:
                log_p_text[i] = 0.6 * log_p_text[i] + 0.4 * log_p_ref[i]
    owl = torch.tensor([s for _, s in cands], device=crop_e.device).clamp_min(1e-3).log()
    return log_p_text + 0.3 * owl


def locate(card, appearances):
    img = card_image(card["id"], card["image"])
    if img is None or not appearances:
        return []
    cands = propose(img)
    if not cands:
        return []
    W, H = img.size
    score = score_matrix(img, cands, appearances)
    rows, cols = linear_sum_assignment(-score.cpu().numpy())
    results = []
    for r, c in zip(rows, cols):
        (x0, y0, x1, y1), _ = cands[c]
        results.append((appearances[r]["appearance_id"], x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H,
                        float(score[r, c].exp())))
    return results


def targets(conn, entity=None, card=None, redo=False, cameos_only=False):
    where, args = ["c.image IS NOT NULL"], []
    if cameos_only:
        where.append("c.id IN (SELECT card_id FROM appearances WHERE role='cameo')")
    if entity:
        where.append("c.id IN (SELECT a2.card_id FROM appearances a2 JOIN entities e2 ON e2.id=a2.entity_id "
                     "WHERE e2.name = ? COLLATE NOCASE)")
        args.append(entity)
    if card:
        where.append("c.id = ?")
        args.append(card)
    # Cards with cameos first: they're the reason this database exists.
    sql = ("SELECT c.id, c.image FROM cards c JOIN appearances a ON a.card_id=c.id WHERE " + " AND ".join(where)
           + " GROUP BY c.id ORDER BY MAX(a.role='cameo') DESC, c.id")
    for c in conn.execute(sql, args).fetchall():
        apps = conn.execute(
            "SELECT a.id AS appearance_id, e.kind, e.name, e.dex, "
            "  (SELECT COUNT(*) FROM boxes b WHERE b.appearance_id=a.id) AS n_boxes, "
            "  (SELECT COUNT(*) FROM boxes b WHERE b.appearance_id=a.id AND (b.verified OR b.source='manual')) AS n_locked "
            "FROM appearances a JOIN entities e ON e.id=a.entity_id WHERE a.card_id=?", (c["id"],)
        ).fetchall()
        todo = [dict(a) for a in apps if not a["n_locked"] and (redo or not a["n_boxes"])]
        if todo:
            yield dict(c), todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", help="entity name, e.g. Cyndaquil or Ethan")
    ap.add_argument("--card", help="TCGdex card id")
    ap.add_argument("--all", action="store_true", help="every card with a known appearance (cameo cards first)")
    ap.add_argument("--cameos", action="store_true", help="only cards with at least one cameo")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--redo", action="store_true", help="replace existing auto boxes")
    args = ap.parse_args()
    if not (args.entity or args.card or args.all or args.cameos):
        ap.error("pass --entity, --card, --cameos or --all")
    conn = connect()
    jobs = list(targets(conn, args.entity, args.card, args.redo, args.cameos))[: args.limit]
    print(f"{len(jobs)} cards to process on {DEVICE}", flush=True)
    # Download images a little ahead of the GPU so the network never stalls inference.
    t0 = time.time()
    loaded = prefetched(lambda job: card_image(job[0]["id"], job[0]["image"]), jobs)
    for i, ((card, apps), _img) in enumerate(loaded, 1):
        try:
            boxes = locate(card, apps)
        except Exception as e:  # one bad image shouldn't kill an hours-long batch
            print(f"[{i}/{len(jobs)}] {card['id']}: ERROR {e!r}", flush=True)
            continue
        with conn:
            conn.executemany("DELETE FROM boxes WHERE appearance_id=? AND source='auto'",
                             [(a["appearance_id"],) for a in apps])
            conn.executemany(
                "INSERT INTO boxes (appearance_id, x, y, w, h, score, source) VALUES (?,?,?,?,?,?,'auto')", boxes
            )
        if i % 50 == 0 or i == len(jobs):
            rate = i / (time.time() - t0)
            print(f"[{i}/{len(jobs)}] {rate:.2f} cards/s, eta {(len(jobs) - i) / rate / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
