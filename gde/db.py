import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB_PATH = DATA / "gde.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sets (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  serie TEXT,
  release_date TEXT
);

CREATE TABLE IF NOT EXISTS cards (
  id TEXT PRIMARY KEY,
  set_id TEXT NOT NULL REFERENCES sets(id),
  local_id TEXT NOT NULL,
  name TEXT NOT NULL,
  category TEXT,
  rarity TEXT,
  illustrator TEXT,
  image TEXT
);
CREATE INDEX IF NOT EXISTS cards_set ON cards(set_id, local_id);
CREATE INDEX IF NOT EXISTS cards_illustrator ON cards(illustrator);

-- Anything that can appear in card art: a Pokémon species or a human character.
CREATE TABLE IF NOT EXISTS entities (
  id TEXT PRIMARY KEY,            -- 'species:155' or 'character:ethan'
  kind TEXT NOT NULL CHECK (kind IN ('species', 'character')),
  name TEXT NOT NULL,
  dex INTEGER,
  generation INTEGER,
  family INTEGER,                 -- PokeAPI evolution_chain_id
  region TEXT
);
CREATE INDEX IF NOT EXISTS entities_name ON entities(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS entities_family ON entities(family);

-- The relationship edge: entity appears on card.
CREATE TABLE IF NOT EXISTS appearances (
  id INTEGER PRIMARY KEY,
  card_id TEXT NOT NULL REFERENCES cards(id),
  entity_id TEXT NOT NULL REFERENCES entities(id),
  role TEXT NOT NULL CHECK (role IN ('main', 'cameo')),
  source TEXT NOT NULL,           -- tcgdex | rotomamiti | name-match | manual
  notes TEXT,
  UNIQUE (card_id, entity_id)
);
CREATE INDEX IF NOT EXISTS appearances_entity ON appearances(entity_id);

-- Where in the card art an appearance is. Coordinates normalized to [0,1].
CREATE TABLE IF NOT EXISTS boxes (
  id INTEGER PRIMARY KEY,
  appearance_id INTEGER NOT NULL REFERENCES appearances(id) ON DELETE CASCADE,
  x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
  score REAL,
  source TEXT NOT NULL,           -- auto | manual
  verified INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS boxes_appearance ON boxes(appearance_id);

-- Your collection: copies you own of a card.
CREATE TABLE IF NOT EXISTS collection (
  card_id TEXT PRIMARY KEY REFERENCES cards(id),
  qty INTEGER NOT NULL DEFAULT 1 CHECK (qty > 0),
  notes TEXT,
  added_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Cameo sheet rows we could not resolve to a TCGdex card (for later alias fixes).
CREATE TABLE IF NOT EXISTS unmatched_cameos (
  entity_id TEXT, card_name TEXT, set_name TEXT, number TEXT, notes TEXT, reason TEXT
);
"""


def connect() -> sqlite3.Connection:
    DATA.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn
