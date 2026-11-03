# Gotta Detect 'Em All

Personal Pokémon TCG database: search a Pokémon or character and see every card it appears on
(as the main subject or as a cameo), where it is in the art, and what it commonly appears with.

## Setup

```sh
uv sync
uv run python -m gde.ingest_species   # ~1,000 species + evolution families (PokeAPI)
uv run python -m gde.ingest_cards     # ~23.7k English cards (TCGdex, cached in data/tcgdex/)
uv run python -m gde.ingest_cameos    # cameo Pokémon + characters (RotomAmiti's sheet)
uv run python -m gde.detect --cameos             # place boxes on cards with cameos (~25 min)
uv run python -m gde.detect --all                # ...then every card (~5 h on an M4 Pro)
uv run python -m gde.train_probe                 # learn species from those boxes
uv run python -m gde.evaluate en-cameo --probe   # open-set test (also: ja-151)
uv run python -m gde.discover                    # suggest cameos no list has; review them in the UI
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

## Open-set identification

`evaluate` asks the harder question — which species are on a card, *without* being told — over all
~1,025 species. A linear probe trained on the pseudo-labeled boxes (`train_probe`) carries most of the gain:

| | en-cameo (145 cards) | | ja-151 (191 Japanese cards) | |
|---|---|---|---|---|
| | baseline | probe v1 | baseline | probe v1 |
| cameo@5 | 46.5% | 66.8% | — | — |
| main@1 | 78.6% | 77.4% | 75.5% | 91.0% |
| recall@5 | 58.1% | 75.7% | 94.8% | 98.4% |

English 151 is excluded from training so ja-151 never sees its art. `discover` turns confident (p ≥ 0.3)
identifications that aren't in any list into suggestions you accept or reject in the card view.
