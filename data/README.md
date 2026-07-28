# Data directory

Datasets, 2D observations, initial guesses, generated masks, pose banks, and
model checkpoints are not distributed in this repository.

Expected layouts are described in `docs/DATA_PREPARATION.md`. Every generated
or downloaded file below this directory is ignored by Git.

The only tracked subdirectory is `protocols/`. It contains the redistributable
condition table for the EMS Occlusion Protocol Suite and no source-dataset
content. See `docs/DATASET_CARD.md`.
