# ABMAP upstream replication status

This repository documents the current state of trying to reproduce the
[rs239/abmap](https://github.com/rs239/abmap) project inside this offline
execution environment.  The original codebase depends on assets that are not
publicly mirrored here, so this project no longer ships a stand-in training
loop with synthetic data.  Instead, it records what is actually available from
upstream (based on the public repository contents and documentation) and how to
integrate it once you have access to the official resources.

## What is available upstream?

The public `rs239/abmap` repository publishes trained checkpoints and inference
utilities, but it does **not** contain the end-to-end training script or the raw
training dataset used in the paper.  The maintainers note that those materials
are available only upon request.  Without them, a faithful reproduction of the
published training run cannot be provided or executed in this environment.

## How to proceed when you obtain the official assets

1. Clone the upstream repository next to this project:
   ```bash
   git clone https://github.com/rs239/abmap.git upstream-abmap
   ```
2. Contact the authors to obtain the training dataset and the private training
   scripts.  Place them inside `upstream-abmap/` following the directory layout
   they provide.
3. Run the orchestration utility that comes with those private assets (for
   example, a `train.py` entry point) directly from within the upstream
   repository.  Until the files are supplied, any attempt to launch a training
   session will result in a missing-file error.

## Repository contents

At this point the repository only tracks this README so that downstream users
are not misled into thinking that a working replica—including the actual
training data—exists here.  Once the official training code and dataset become
public, they can be added as a submodule or vendored in a follow-up change.
