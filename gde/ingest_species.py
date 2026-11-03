"""Load every Pokémon species (English name, generation, evolution family) from PokeAPI's CSV dump."""
import csv
import io

import httpx

from gde.db import connect

BASE = "https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv"
ENGLISH = "9"


def fetch_csv(name):
    r = httpx.get(f"{BASE}/{name}.csv", timeout=60, follow_redirects=True)
    r.raise_for_status()
    return list(csv.DictReader(io.StringIO(r.text)))


def main():
    species = fetch_csv("pokemon_species")
    names = {
        row["pokemon_species_id"]: row["name"]
        for row in fetch_csv("pokemon_species_names")
        if row["local_language_id"] == ENGLISH
    }
    rows = [
        (f"species:{s['id']}", "species", names.get(s["id"], s["identifier"].title()),
         int(s["id"]), int(s["generation_id"]), int(s["evolution_chain_id"] or 0) or None)
        for s in species
    ]
    conn = connect()
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO entities (id, kind, name, dex, generation, family) VALUES (?,?,?,?,?,?)", rows
        )
    print(f"loaded {len(rows)} species")


if __name__ == "__main__":
    main()
