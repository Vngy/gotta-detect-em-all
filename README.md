# Gotta Detect 'Em All

Personal Pokémon TCG database: search a Pokémon or character and see every card it appears on
(as the main subject or as a cameo), where it is in the art, and what it commonly appears with.

## Setup

```sh
uv sync
uv run python -m gde.ingest_species   # ~1,000 species + evolution families (PokeAPI)
uv run python -m gde.ingest_cards     # ~23.7k English cards (TCGdex, cached in data/tcgdex/)
uv run python -m gde.ingest_cameos    # cameo Pokémon + characters (RotomAmiti's sheet)
uv run python -m gde.detect --entity Cyndaquil   # locate boxes for every card with Cyndaquil
uv run uvicorn gde.api:app --port 8765 --reload  # http://localhost:8765
```

Re-run `ingest_cameos` whenever the sheet updates. Rows that don't match a TCGdex card are kept
in the `unmatched_cameos` table (mostly Japanese-only releases); add set aliases in `SET_ALIASES`.

## Model

- `entities` — species (`species:155`) and characters (`character:ethan`)
- `appearances` — entity ↔ card edge, `role` = main | cameo, with provenance (`source`)
- `boxes` — normalized location of an appearance; `auto` boxes are dashed in the UI,
  hand-drawn or confirmed boxes are solid and never overwritten by `detect`

## Localization

OWLv2 proposes regions; CLIP does zero-shot classification of each region over the card's
*known* entities plus distractor labels, blended with similarity to official artwork; a
one-to-one assignment picks a box per entity. Check quality with
`uv run python scripts/preview_boxes.py out.png <card ids...> --highlight Cyndaquil`.
