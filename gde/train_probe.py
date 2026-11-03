"""Train a species classifier (linear probe on CLIP features) from the boxes detect.py placed.

detect.py only has to *place* a card's known entities, which is easy; the boxes it produces are pseudo-labeled
crops of card art. A probe trained on them teaches open-set identification (evaluate.py) what Pokémon look
like *in card illustrations*, not just in official artwork.

Training data: species boxes that are human-verified/drawn, or auto boxes with score >= MIN_SCORE (spot-checked:
~70% correct in 0.2–0.5, mostly wrong below 0.2), plus one official-artwork sample per species so every class
exists. Held out: the en-cameo eval cards and all of English 151 (sv03.5), whose art the ja-151 eval reuses.

  uv run python -m gde.train_probe
"""
import random
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image, ImageOps

from gde.db import DATA, connect
from gde.detect import DEVICE, IMAGES, clip_image_emb, padded_crop, reference_emb
from gde.evaluate import en_cameo

PROBE = DATA / "probe.pt"
FEATURES = DATA / "probe_features.pt"
MIN_SCORE = 0.2
HELDOUT_SETS = {"sv03.5"}


def training_boxes(conn):
    heldout = {it["id"] for it in en_cameo(150, 0)}
    rows = conn.execute(
        """
        SELECT a.card_id, c.set_id, e.dex, b.x, b.y, b.w, b.h
        FROM boxes b JOIN appearances a ON a.id = b.appearance_id
        JOIN entities e ON e.id = a.entity_id JOIN cards c ON c.id = a.card_id
        WHERE e.kind = 'species' AND (b.verified = 1 OR b.source = 'manual' OR b.score >= ?)
        """, (MIN_SCORE,),
    ).fetchall()
    return [r for r in rows if r["card_id"] not in heldout and r["set_id"] not in HELDOUT_SETS]


def crops_for(card_id, boxes):
    path = IMAGES / f"{card_id}.webp"
    if not path.exists():
        return []
    img = Image.open(path).convert("RGB")
    W, H = img.size
    out = []
    for r in boxes:
        c = padded_crop(img, (r["x"] * W, r["y"] * H, (r["x"] + r["w"]) * W, (r["y"] + r["h"]) * H))
        out += [(c, r["dex"], card_id), (ImageOps.mirror(c), r["dex"], card_id)]
    return out


def extract(conn):
    boxes = training_boxes(conn)
    by_card = {}
    for r in boxes:
        by_card.setdefault(r["card_id"], []).append(r)
    print(f"{len(boxes)} training boxes on {len(by_card)} cards")
    feats, labels, groups, batch = [], [], [], []

    def flush():
        if batch:
            feats.append(clip_image_emb([c for c, _, _ in batch]).cpu())
            labels.extend(d for _, d, _ in batch)
            groups.extend(g for _, _, g in batch)
            batch.clear()

    with ThreadPoolExecutor(8) as pool:
        for i, crops in enumerate(pool.map(lambda kv: crops_for(*kv), by_card.items()), 1):
            for item in crops:
                batch.append(item)
                if len(batch) == 64:
                    flush()
            if i % 1000 == 0:
                print(f"  {i}/{len(by_card)} cards embedded", flush=True)
    flush()
    X, y = torch.cat(feats), torch.tensor(labels)
    torch.save({"X": X, "y": y, "groups": groups}, FEATURES)
    return X, y, groups


def train(X, y, groups, epochs=60, lr=1e-3, wd=1e-4):
    conn = connect()
    dexes = [r[0] for r in conn.execute("SELECT dex FROM entities WHERE kind='species' ORDER BY dex")]
    index = {d: i for i, d in enumerate(dexes)}
    refs = [(reference_emb(d), d) for d in dexes]
    ref_X = torch.stack([r.cpu() for r, _ in refs if r is not None])
    ref_y = torch.tensor([d for r, d in refs if r is not None])
    # Validation: 5% of *cards* (so a crop's mirror never straddles the split); artwork stays in training.
    cards = sorted(set(groups))
    random.Random(0).shuffle(cards)
    val_cards = set(cards[: max(1, len(cards) // 20)])
    is_val = torch.tensor([g in val_cards for g in groups] + [False] * (2 * len(ref_y)))
    X, y = torch.cat([X, ref_X, ref_X]), torch.cat([y, ref_y, ref_y])  # artwork counted twice: anchors rare classes
    t = torch.tensor([index[int(d)] for d in y])
    val, tr = is_val.nonzero().squeeze(1), (~is_val).nonzero().squeeze(1)
    model = torch.nn.Linear(X.shape[1], len(dexes)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xd, td = X.to(DEVICE), t.to(DEVICE)
    for epoch in range(epochs):
        model.train()
        for idx in tr[torch.randperm(len(tr))].split(512):
            loss = torch.nn.functional.cross_entropy(model(Xd[idx] * 20), td[idx], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 9 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(Xd[val] * 20).argmax(-1) == td[val]).float().mean()
            print(f"  epoch {epoch + 1}: loss {loss.item():.3f}, held-out-card crop acc {acc:.1%}", flush=True)
    torch.save({"dexes": dexes, "state": model.cpu().state_dict()}, PROBE)
    print(f"saved {PROBE}")


def main():
    random.seed(0)
    conn = connect()
    X, y, groups = extract(conn)
    print(f"{len(X)} crop embeddings (with flips), {len(set(y.tolist()))} species")
    train(X, y, groups)


if __name__ == "__main__":
    main()
