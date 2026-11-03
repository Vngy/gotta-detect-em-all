"""Fetch every English card from TCGdex (cached to data/tcgdex/) and load sets + cards + main-Pokémon edges."""
import asyncio
import json

import httpx

from gde.db import DATA, connect

API = "https://api.tcgdex.net/v2/en"
CACHE = DATA / "tcgdex"
CONCURRENCY = 16


async def fetch_json(client: httpx.AsyncClient, path: str, cache_file):
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    for attempt in range(4):
        try:
            r = await client.get(f"{API}/{path}")
            if r.status_code == 404:
                return None
            r.raise_for_status()
            cache_file.write_text(r.text)
            return r.json()
        except httpx.HTTPError:
            await asyncio.sleep(2**attempt)
    raise RuntimeError(f"failed to fetch {path}")


async def fetch_all():
    (CACHE / "sets").mkdir(parents=True, exist_ok=True)
    (CACHE / "cards").mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "gotta-detect-em-all/0.1"}) as client:
        set_list = await fetch_json(client, "sets", CACHE / "sets.json")

        async def one(path, file):
            async with sem:
                return await fetch_json(client, path, file)

        sets = await asyncio.gather(*(one(f"sets/{s['id']}", CACHE / "sets" / f"{s['id']}.json") for s in set_list))
        sets = [s for s in sets if s]
        card_ids = [c["id"] for s in sets for c in s.get("cards", [])]
        print(f"{len(sets)} sets, {len(card_ids)} cards")
        done = 0

        async def card(cid):
            nonlocal done
            res = await one(f"cards/{cid}", CACHE / "cards" / f"{cid.replace('/', '_')}.json")
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(card_ids)}")
            return res

        cards = await asyncio.gather(*(card(cid) for cid in card_ids))
    return sets, [c for c in cards if c]


def load(sets, cards):
    conn = connect()
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO sets VALUES (?,?,?,?)",
            [(s["id"], s["name"], (s.get("serie") or {}).get("name"), s.get("releaseDate")) for s in sets],
        )
        conn.executemany(
            "INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?,?,?,?)",
            [
                (c["id"], c["set"]["id"], c["localId"], c["name"], c.get("category"),
                 c.get("rarity"), c.get("illustrator"), c.get("image"))
                for c in cards
            ],
        )
        known = {r[0] for r in conn.execute("SELECT dex FROM entities WHERE kind='species'")}
        edges = [
            (c["id"], f"species:{dex}")
            for c in cards
            for dex in (c.get("dexId") or [])
            if dex in known
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO appearances (card_id, entity_id, role, source) VALUES (?,?,'main','tcgdex')",
            edges,
        )
    print(f"loaded {len(sets)} sets, {len(cards)} cards, {len(edges)} main-Pokémon edges")


def main():
    sets, cards = asyncio.run(fetch_all())
    load(sets, cards)


if __name__ == "__main__":
    main()
