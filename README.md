# Whisker

Whisker trains its chess model from rated PGN games and provides a local web
app and UCI engine. The policy learns to choose recorded moves; the value head
learns game results. During play, the engine selects its highest-scoring legal
move without searching ahead.

## Play locally

```bash
python -m pip install -r requirements.txt
python -m src.serve --checkpoint checkpoints/your-checkpoint.pt
```

Open <http://127.0.0.1:8000>. Without a checkpoint, the app opens in preview
mode. It supports bullet, blitz, and rapid controls. A checkpoint in
`checkpoints/` can be selected automatically, or choose one with
`--checkpoint`.

## Train

Download and parse monthly archives into four rating bands through 2200 Elo, then train a base model:

```bash
python -m src.download_months --year 2024
# Limit this run to one month from the selected range:
python -m src.download_months --year 2024 --start-month 1 --end-month 12 --month-count 1
python -m src.train --data data/processed/elo-1000-1200 \
  --output-dir checkpoints/elo1000 --steps 50000
# Defaults to batch size 512 and zero DataLoader subprocesses:
BATCH_SIZE=1024 bash train_curriculum.sh
# Set WORKERS above zero only if you want parallel data loading:
WORKERS=4 bash train_curriculum.sh
```

While one month is being parsed, the downloader fetches the next needed archive.
It reuses/resumes temporary archives and removes each after all four band files
are complete. Band limits are inclusive.

Fine-tune on a stronger band with optional replay from the base set using
`python -m src.train --fine-tune CHECKPOINT --replay-data DATA --replay-ratio 0.25`.
Use `--resume` to continue an interrupted run; fine-tuning starts a new stage.

## Evaluate

Run `python -m src.engine --checkpoint CHECKPOINT` as a UCI engine for a
quick smoke test. To estimate playing strength locally, install
`cutechess-cli` and Stockfish, then run:

```bash
python -m src.benchmark --checkpoint CHECKPOINT --stockfish stockfish \
  --elos 1500 1800 2100 --games 100 --tc 60+0.6
```

The benchmark plays both colors from varied openings and saves PGNs under
`benchmark-results/`. Its Elo estimate depends on the Stockfish build and time
control; it is not an online rating.

## Third-party assets

The MP3 sounds and the profile king SVGs in `src/web/assets/` come from
[WintrChess](https://github.com/WintrCat/wintrchess), specifically
`client/public/audio/` and `client/public/img/pieces/standard/`. WintrChess is
licensed under GPL-3.0; its license is included at
[`src/web/assets/WINTRCHESS-GPL-3.0.txt`](src/web/assets/WINTRCHESS-GPL-3.0.txt).

The Cburnett board-piece SVGs in `src/web/assets/pieces/cburnett/` are
reformatted exports of the default piece art bundled by WintrChess's pinned
`react-chessboard` 4.7.3 dependency. The art is by Colin M. L. Burnett
(Cburnett), adapted from the
[Wikimedia Commons SVG chess pieces](https://commons.wikimedia.org/wiki/Category:SVG_chess_pieces)
and distributed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
