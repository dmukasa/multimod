# ABMAP training reproduction toolkit

This repository reverse engineers the end-to-end training loop that powers the
`rs239/abmap` project by following the experimental description in the
[PNAS article (doi:10.1073/pnas.2418918121)](https://www.pnas.org/doi/10.1073/pnas.2418918121).
It provides utilities to fetch the publicly released supplementary datasets,
transform them into a machine-learning ready format, and train the paper's
sequence-to-binding regression model on the full cohort.

> **Note:** The download helpers use the officially published supplementary file
> URLs.  If you run this code from a network that cannot reach `pnas.org`,
> manually download the files listed in `configs/abmap_full.yaml` and place them
> inside `data/raw/` before running the preprocessing step.

## Repository layout

- `configs/abmap_full.yaml` – hyperparameters inferred from the Methods section
  and supplementary material, covering sequence lengths, transformer depth, and
  optimizer settings.
- `scripts/download_abmap_dataset.py` – pulls the supplementary Excel workbooks
  referenced in the paper.
- `scripts/prepare_abmap_dataset.py` – parses the raw tables and writes the
  processed Parquet dataset used by the trainer.
- `scripts/train_abmap.py` – launches the PyTorch training loop that mirrors the
  paper's heavy/light chain + antigen encoder architecture.
- `src/abmap/` – Python package that implements configuration handling, data
  tokenisation, model components, and the training engine.
- `notebooks/abmap_training.ipynb` – an executable walkthrough that downloads
  the data, builds the dataset, trains the model, and visualises training
  metrics.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 1. Download the supplementary datasets

```bash
python scripts/download_abmap_dataset.py data/raw
```

This command fetches the two supplementary Excel files (`sd01` and `sd02`) and
records their SHA-256 checksums so you can verify integrity.

## 2. Build the processed training table

```bash
python scripts/prepare_abmap_dataset.py \
  data/raw/pnas.2418918121.sd01.xlsx \
  data/raw/pnas.2418918121.sd02.xlsx \
  --output data/processed/abmap_training.parquet
```

The parser uses schema heuristics to recover the heavy-chain, light-chain,
antigen, and binding columns automatically.  If a future revision of the
supplementary files changes column names, adjust `configs/abmap_full.yaml` or
extend the heuristics in `src/abmap/data/parsing.py`.

## 3. Train the full ABMAP model

```bash
python scripts/train_abmap.py configs/abmap_full.yaml --output-dir artifacts/full
```

The trainer replicates the architecture described in the publication:
transformer encoders embed the heavy, light, and antigen sequences, their
representations interact multiplicatively, and a multi-layer regression head
predicts the binding score.  Checkpoints and a JSON training history are written
under `artifacts/full/`.

## 4. Interactive notebook

Open `notebooks/abmap_training.ipynb` to run the entire workflow in an
interactive environment.  The notebook mirrors the CLI commands above and
includes cells to inspect the dataset, monitor learning curves, and export the
best-performing checkpoint.

## Troubleshooting

- **Column inference failed:** Inspect the raw Excel sheets and add explicit
  column overrides to `src/abmap/data/parsing.py` or pre-process the files
  manually.
- **Download blocked:** Fetch the supplementary datasets in a browser and copy
  them into `data/raw/`.
- **CUDA not available:** Set `training.device` to `cpu` in the YAML config or
  export `ABMAP_DEVICE=cpu` and override via `--device` flag in your launcher.
