"""Search API + static UI.  Run: uv run uvicorn gde.api:app --reload"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from gde.db import ROOT, connect

app = FastAPI(title="Gotta Detect 'Em All")
WEB = ROOT / "web"

CARD_COLS = "(SELECT qty FROM collection WHERE card_id = c.id) AS owned, c.id, c.name, c.local_id, c.rarity, c.illustrator, c.image, c.category, s.id AS set_id, s.name AS set_name, s.serie, s.release_date"


def rows(sql, *args):
    conn = connect()
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def boxes_for(appearance_ids):
    if not appearance_ids:
        return {}
    q = ",".join("?" * len(appearance_ids))
    out = {}
    for b in rows(f"SELECT * FROM boxes WHERE appearance_id IN ({q})", *appearance_ids):
        out.setdefault(b["appearance_id"], []).append(b)
    return out


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/search")
def search(q: str, limit: int = 12):
    q = q.strip()
    if not q:
        return []
    return rows(
        """
        SELECT e.id, e.kind, e.name, e.dex, e.region,
               SUM(a.role='main') AS n_main, SUM(a.role='cameo') AS n_cameo
        FROM entities e LEFT JOIN appearances a ON a.entity_id = e.id
        WHERE e.name LIKE ? COLLATE NOCASE OR CAST(e.dex AS TEXT) = ?
        GROUP BY e.id
        ORDER BY (e.name LIKE ? COLLATE NOCASE) DESC, COUNT(a.id) DESC
        LIMIT ?
        """,
        f"%{q}%", q, f"{q}%", limit,
    )


@app.get("/api/entity/{entity_id}")
def entity(entity_id: str):
    found = rows("SELECT * FROM entities WHERE id = ?", entity_id)
    if not found:
        raise HTTPException(404)
    ent = found[0]
    cards = rows(
        f"""
        SELECT {CARD_COLS}, a.id AS appearance_id, a.role, a.notes, a.source
        FROM appearances a JOIN cards c ON c.id = a.card_id JOIN sets s ON s.id = c.set_id
        WHERE a.entity_id = ?
        ORDER BY a.role = 'main', s.release_date, c.set_id, CAST(c.local_id AS INTEGER)
        """,
        entity_id,
    )
    bx = boxes_for([c["appearance_id"] for c in cards])
    for c in cards:
        c["boxes"] = bx.get(c["appearance_id"], [])

    co = rows(
        """
        SELECT e.id, e.kind, e.name, COUNT(*) AS n
        FROM appearances a1
        JOIN appearances a2 ON a2.card_id = a1.card_id AND a2.entity_id != a1.entity_id
        JOIN entities e ON e.id = a2.entity_id
        WHERE a1.entity_id = ?
        GROUP BY e.id ORDER BY n DESC LIMIT 30
        """,
        entity_id,
    )
    illustrators = rows(
        """
        SELECT c.illustrator AS name, COUNT(*) AS n, SUM(a.role='cameo') AS n_cameo
        FROM appearances a JOIN cards c ON c.id = a.card_id
        WHERE a.entity_id = ? AND c.illustrator IS NOT NULL
        GROUP BY c.illustrator ORDER BY n DESC LIMIT 15
        """,
        entity_id,
    )
    family = []
    if ent["family"]:
        family = rows(
            """
            SELECT e.id, e.name, e.dex, COUNT(a.id) AS n
            FROM entities e LEFT JOIN appearances a ON a.entity_id = e.id
            WHERE e.family = ? GROUP BY e.id ORDER BY e.dex
            """,
            ent["family"],
        )
    return {
        "entity": ent,
        "cards": cards,
        "relations": {
            "co_appears_with": co,
            "illustrators": illustrators,
            "evolution_family": family,
        },
    }


@app.get("/api/card/{card_id}")
def card(card_id: str):
    found = rows(f"SELECT {CARD_COLS} FROM cards c JOIN sets s ON s.id = c.set_id WHERE c.id = ?", card_id)
    if not found:
        raise HTTPException(404)
    apps = rows(
        """
        SELECT a.id AS appearance_id, a.role, a.notes, a.source, e.id AS entity_id, e.kind, e.name
        FROM appearances a JOIN entities e ON e.id = a.entity_id WHERE a.card_id = ?
        ORDER BY a.role DESC, e.name
        """,
        card_id,
    )
    bx = boxes_for([a["appearance_id"] for a in apps])
    for a in apps:
        a["boxes"] = bx.get(a["appearance_id"], [])
    return {"card": found[0], "appearances": apps}


class BoxIn(BaseModel):
    appearance_id: int
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)


@app.post("/api/boxes")
def add_box(b: BoxIn):
    """A hand-drawn box replaces any auto box for that appearance and is locked against re-detection."""
    conn = connect()
    with conn:
        conn.execute("DELETE FROM boxes WHERE appearance_id = ? AND source = 'auto'", (b.appearance_id,))
        cur = conn.execute(
            "INSERT INTO boxes (appearance_id, x, y, w, h, source, verified) VALUES (?,?,?,?,?,'manual',1)",
            (b.appearance_id, b.x, b.y, b.w, b.h),
        )
    return {"id": cur.lastrowid}


@app.post("/api/boxes/{box_id}/verify")
def verify_box(box_id: int):
    conn = connect()
    with conn:
        conn.execute("UPDATE boxes SET verified = 1 WHERE id = ?", (box_id,))
    return {"ok": True}


@app.delete("/api/boxes/{box_id}")
def delete_box(box_id: int):
    conn = connect()
    with conn:
        conn.execute("DELETE FROM boxes WHERE id = ?", (box_id,))
    return {"ok": True}


class OwnedIn(BaseModel):
    qty: int = Field(ge=0, le=999)


@app.put("/api/collection/{card_id}")
def set_owned(card_id: str, body: OwnedIn):
    """qty 0 removes the card from your collection."""
    conn = connect()
    with conn:
        if body.qty == 0:
            conn.execute("DELETE FROM collection WHERE card_id = ?", (card_id,))
        else:
            conn.execute(
                "INSERT INTO collection (card_id, qty) VALUES (?, ?) "
                "ON CONFLICT(card_id) DO UPDATE SET qty = excluded.qty", (card_id, body.qty),
            )
    return {"card_id": card_id, "qty": body.qty}
