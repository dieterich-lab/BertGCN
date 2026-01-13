#!/bin/bash
# Safe cleanup for outputs/gcn: move older ephemeral folders into a tidy mlruns hierarchy.
# Skips folders modified within the last 10 minutes to avoid interfering with active runs.

# This script is the tracked copy of the cleanup helper (a runtime copy exists under slurm/).

ROOT=/beegfs/homes/pwiesenbach/BertGCN
GDIR=$ROOT/outputs/gcn
MLRUNS=$GDIR/mlruns
NOW=$(date +%s)
SAFE_SEC=$((10*60))

mkdir -p "$MLRUNS/multirun"
mkdir -p "$MLRUNS/sweeps"
mkdir -p "$MLRUNS/archive"

mv_if_old() {
  src=$1
  dest=$2
  if [ ! -e "$src" ]; then
    return
  fi
  for entry in "$src"/*; do
    [ -e "$entry" ] || continue
    mtime=$(stat -c %Y "$entry")
    age=$((NOW-mtime))
    base=$(basename "$entry")
    if [ $age -ge $SAFE_SEC ]; then
      echo "Moving $entry -> $dest/$base"
      mv "$entry" "$dest/"
    else
      echo "Skipping recent $entry (age ${age}s)"
    fi
  done
}

echo "Starting tidy of $GDIR -> canonical mlruns layout"

if [ -d "$GDIR/multirun" ]; then
  mv_if_old "$GDIR/multirun" "$MLRUNS/multirun"
  if [ -d "$GDIR/multirun" ] && [ -z "$(ls -A "$GDIR/multirun")" ]; then
    rmdir "$GDIR/multirun"
  fi
fi

shopt -s nullglob
for s in "$GDIR"/sweeps*; do
  [ -e "$s" ] || continue
  mv_if_old "$s" "$MLRUNS/sweeps"
  if [ -d "$s" ] && [ -z "$(ls -A "$s")" ]; then
    rmdir "$s"
  fi
done
shopt -u nullglob

for f in "$GDIR"/*; do
  [ -e "$f" ] || continue
  name=$(basename "$f")
  case "$name" in
    mlruns|downloads) continue ;;
    multirun|sweeps*) continue ;;
    *)
      if [ -d "$f" ]; then
        mtime=$(stat -c %Y "$f")
        age=$((NOW-mtime))
        if [ $age -ge $SAFE_SEC ]; then
          echo "Archiving $f -> $MLRUNS/archive/"
          mv "$f" "$MLRUNS/archive/"
        else
          echo "Skipping recent $f (age ${age}s)"
        fi
      fi
      ;;
  esac
done

echo "Cleanup complete. Please verify outputs/gcn/mlruns contains expected subfolders."
