#!/bin/bash
# TASK-024 finals chain, strictly in the order pre-registered in docs/experiments/2026-10-05_truck_scale_study/README.md (amendment 5)
PY=C:/v/e/Scripts/python.exe
DL=2026-10-07T12:40
$PY scripts/run_truck_scale_sweep.py final --config E1 --workers 1 --deadline $DL
$PY scripts/run_baseline_warmup_check.py --seeds 2000-2002 --warmup 0.05 --deadline $DL
for c in S2 S3 S1; do
  $PY scripts/run_truck_scale_sweep.py final --config $c --workers 1 --deadline $DL
done
