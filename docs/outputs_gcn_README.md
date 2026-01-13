Canonical layout for outputs/gcn

This document mirrors the README placed inside the outputs directory, but
keeps a tracked copy in `docs/` so the project can version-control the
expected layout without committing ephemeral output files.

Layout:

- mlruns/             # canonical MLflow root for GCN experiments
  - multirun/         # hydra multirun output directories (timestamped)
  - sweeps/           # sweep run outputs (one folder per sweep)
  - archive/          # older or miscellaneous ephemeral data

Use `scripts/cleanup_outputs_gcn.sh` to safely reorganize `outputs/gcn` into
this structure. The cleanup script skips folders modified within the last
10 minutes to avoid disrupting active runs.
