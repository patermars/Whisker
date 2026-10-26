# Whisker

Whisker trains a ChessNet model from rated PGN games and provides a local web
app and UCI engine. The policy learns to choose recorded moves; the value head
learns game results. During play, the engine selects its highest-scoring legal
move without searching ahead.

## Play locally

```bash
python -m pip install -r requirements.txt
python -m whisker.serve --checkpoint checkpoints/your-checkpoint.pt
```

Open <http://127.0.0.1:8000>. Without a checkpoint, the app opens in preview
mode. It supports bullet, blitz, and rapid controls. A checkpoint in
`checkpoints/` can be selected automatically, or choose one with
`--checkpoint`.

## Train

Download monthly games in a rating band, then train a base model:

```bash
python -m whisker.download_months --year 2024 --min-elo 1000 --max-elo 1099
python -m whisker.train --data data/processed/elo-1000-1099 \
  --output-dir checkpoints/elo1000 --steps 50000
```

Fine-tune on a stronger band with optional replay from the base set using
`python -m whisker.train --fine-tune CHECKPOINT --replay-data DATA --replay-ratio 0.25`.
Use `--resume` to continue an interrupted run; fine-tuning starts a new stage.

## Evaluate

Run `python -m whisker.engine --checkpoint CHECKPOINT` as a UCI engine for a
quick smoke test. To estimate playing strength locally, install
`cutechess-cli` and Stockfish, then run:

```bash
python -m whisker.benchmark --checkpoint CHECKPOINT --stockfish stockfish \
  --elos 1500 1800 2100 --games 100 --tc 60+0.6
```

The benchmark plays both colors from varied openings and saves PGNs under
`benchmark-results/`. Its Elo estimate depends on the Stockfish build and time
control; it is not an online rating.