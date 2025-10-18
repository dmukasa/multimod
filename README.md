# multimod ABMAP Replica

This repository provides a dependency-free recreation of the core training
loop from the [ABMAP](https://github.com/rs239/abmap) project.  Because the
original codebase relies on external packages and datasets that are not
available in this execution environment, the implementation here uses a
synthetic dataset and a pure-Python logistic regression model to mimic the
training and validation workflow.

## Project layout

```
.
├── abmap/                     # Lightweight data, model, and training utilities
├── data/                      # Synthetic dataset generated for this example
├── notebooks/
│   └── abmap_full_training.ipynb  # Step-by-step notebook covering full training
├── scripts/
│   ├── generate_dataset.py    # Utility script to regenerate the dataset
│   └── train_main_model.py    # Command-line entry point for model training
└── artifacts/                 # Output directory for saved models and metrics
```

## Generate the dataset

```bash
python scripts/generate_dataset.py --samples 4000 --seed 11 --noise 0.5 --output data/main_dataset.csv
```

## Train the main model

```bash
python scripts/train_main_model.py --epochs 30 --batch-size 128 --learning-rate 0.2 --val-ratio 0.25
```

The command prints the final metrics and stores them together with the model
parameters in the `artifacts/` directory.

## Run the notebook

Open `notebooks/abmap_full_training.ipynb` in JupyterLab or VS Code to walk
through the full workflow interactively.  The notebook mirrors the command
line process: it loads the dataset, trains the model on the full corpus,
reviews the recorded metrics, and persists the resulting artifacts.
