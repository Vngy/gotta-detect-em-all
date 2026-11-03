"""Load RotomAmiti's Cameo Pokémon Card Database (Google Sheet) as cameo edges, plus human characters.

The sheet lists, per species, every card it appears on without being the named Pokémon, and per human
character, every card they appear on (except Trainer cards named after them — those we add as 'main').
"""
import csv
import io
import re

import httpx

from gde.db import connect

SHEET = "18nIkOgqQrHZTz0TrH_gL1e1nL1RcHiCmPF5finAjToY"
POKEMON_TABS = ["0", "2112540589", "1642805847", "623394955", "907311085",
                "1750206679", "2096460131", "692781906", "1784063283"]
TRAINER_TAB = "36967854"

# Sheet set name -> TCGdex set name, where they differ. Extend as unmatched_cameos reveals more.
# "XX-P Promos" in the sheet are Japanese promo lines; they are NOT the English "XX Promos".
SET_ALIASES = {
    "xy promos": "xy black star promos",
    "bw promos": "bw black star promos",
    "hgss promos": "hgss black star promos",
    "dp promos": "dp black star promos",
    "nintendo promos": "nintendo black star promos",
    "wizards promos": "wizards black star promos",
    "mep promos": "mep black star promos",
    "sv promos": "svp black star promos",
    "svp promos": "svp black star promos",
    "swsh promos": "swsh black star promos",
    "sm promos": "sm black star promos",
    "heartgold & soulsilver": "heartgold soulsilver",
}


def norm(s: str) -> str:
    s = s.lower().replace("&", "and").replace("é", "e")
    return re.sub(r"[^a-z0-9]", "", s)


def norm_num(n: str) -> str:
    n = n.strip().upper()
    m = re.fullmatch(r"([A-Z]*)0*(\d+)([A-Z]*)", n)
    return f"{m[1]}{int(m[2])}{m[3]}" if m else n


def digits(n: str) -> str | None:
    m = re.search(r"\d+", n)
    return str(int(m[0])) if m else None


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def fetch_tab(gid):
    r = httpx.get(f"https://docs.google.com/spreadsheets/d/{SHEET}/export",
                  params={"format": "csv", "gid": gid}, timeout=60, follow_redirects=True)
    r.raise_for_status()
    return list(csv.reader(io.StringIO(r.text)))


class CardIndex:
    def __init__(self, conn):
        self.by_set_name: dict[str, list[str]] = {}
        for sid, name in conn.execute("SELECT id, name FROM sets"):
            self.by_set_name.setdefault(norm(name), []).append(sid)
        self.cards: dict[tuple[str, str], list[tuple[str, str]]] = {}
        self.by_set: dict[str, list[tuple[str, str]]] = {}
        for cid, sid, lid, name in conn.execute("SELECT id, set_id, local_id, name FROM cards"):
            self.cards.setdefault((sid, norm_num(lid)), []).append((cid, name, lid))
            self.by_set.setdefault(sid, []).append((cid, name, lid))

    def resolve(self, card_name, set_name, number):
        key = norm(SET_ALIASES.get(set_name.strip().lower(), set_name))
        exact = self.by_set_name.get(key, [])
        # Subsets such as "Crown Zenith Galarian Gallery" or "Expedition Base Set" share a name prefix.
        prefixed = [sid for k, ids in self.by_set_name.items() if k != key and k.startswith(key) for sid in ids]
        if not exact and not prefixed:
            return None, "unknown set"
        name = norm(card_name)
        for set_ids in (exact, prefixed):
            hits = [c for sid in set_ids for c in self.cards.get((sid, norm_num(number)), [])]
            if len(hits) > 1:
                hits = [c for c in hits if norm(c[1]) == name] or hits[:1]
            if not hits:  # promo prefixes: sheet "102" vs TCGdex "SM102"
                d = digits(number)
                hits = [c for sid in set_ids for c in self.by_set.get(sid, [])
                        if d and digits(c[2]) == d and norm(c[1]) == name]
            if not hits:  # number missing/odd ("©2001", "-"): name alone, if unambiguous
                hits = [c for sid in set_ids for c in self.by_set.get(sid, []) if norm(c[1]) == name]
                hits = hits if len(hits) == 1 else []
            if hits:
                return hits[0][0], None
        return None, "no card"


def forward_fill_rows(rows, header_rows=1):
    """Yield (col0, col1, card_name, set, number, notes) with merged-cell blanks filled from above."""
    c0 = c1 = card = None
    for row in rows[header_rows:]:
        row = (row + [""] * 6)[:6]
        if not any(row[2:5]):
            continue
        if row[1]:
            c1, card = row[1], None
            c0 = row[0] or c0
        elif row[0]:
            c0 = row[0]
        card = row[2] or card
        yield c0, c1, card, row[3], row[4], row[5]


def main():
    conn = connect()
    idx = CardIndex(conn)
    edges, unmatched, characters = [], [], {}

    for gid in POKEMON_TABS:
        for ndex, _name, card, set_name, num, notes in forward_fill_rows(fetch_tab(gid)):
            m = re.match(r"\d+", ndex or "")
            if not m:
                continue
            ent = f"species:{int(m[0])}"
            cid, why = idx.resolve(card, set_name, num)
            if cid:
                edges.append((cid, ent, notes or None))
            else:
                unmatched.append((ent, card, set_name, num, notes, why))

    for region, who, card, set_name, num, notes in forward_fill_rows(fetch_tab(TRAINER_TAB)):
        if not who:
            continue
        ent = f"character:{slug(who)}"
        characters[ent] = (who, (region or "").title() or None)
        cid, why = idx.resolve(card, set_name, num)
        if cid:
            edges.append((cid, ent, notes or None))
        else:
            unmatched.append((ent, card, set_name, num, notes, why))

    with conn:
        conn.execute("DELETE FROM appearances WHERE source IN ('rotomamiti', 'name-match')")
        conn.execute("DELETE FROM unmatched_cameos")
        conn.executemany(
            "INSERT OR REPLACE INTO entities (id, kind, name, region) VALUES (?, 'character', ?, ?)",
            [(e, n, r) for e, (n, r) in characters.items()],
        )
        conn.executemany(
            "INSERT OR IGNORE INTO appearances (card_id, entity_id, role, source, notes) "
            "VALUES (?,?,'cameo','rotomamiti',?)", edges,
        )
        # Trainer cards named after a character feature that character as the main subject.
        names = sorted(characters.items(), key=lambda kv: -len(kv[1][0]))
        mains = []
        for cid, cname in conn.execute("SELECT id, name FROM cards WHERE category='Trainer'"):
            for ent, (who, _) in names:
                if len(who) > 3 and re.search(rf"\b{re.escape(who)}\b", cname):
                    mains.append((cid, ent))
                    break
        conn.executemany(
            "INSERT OR IGNORE INTO appearances (card_id, entity_id, role, source) VALUES (?,?,'main','name-match')",
            mains,
        )
        conn.executemany("INSERT INTO unmatched_cameos VALUES (?,?,?,?,?,?)", unmatched)

    total = len(edges) + len(unmatched)
    print(f"cameo rows: {total}, matched {len(edges)} ({len(edges) / max(total, 1):.0%}), "
          f"characters: {len(characters)}, trainer-main edges: {len(mains)}")
    top = {}
    for _, _, s, *_rest, why in unmatched:
        if why == "unknown set":
            top[s] = top.get(s, 0) + 1
    print("top unknown sets:", sorted(top.items(), key=lambda kv: -kv[1])[:25])


if __name__ == "__main__":
    main()
