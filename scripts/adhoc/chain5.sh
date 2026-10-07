#!/bin/bash
PY=C:/v/e/Scripts/python.exe
DL=2026-10-08T07:00
$PY scripts/run_truck_scale_sweep.py final --config E1 --seeds 72-81 --workers 1 --deadline $DL
$PY scripts/run_baseline_warmup_check.py final --configs G4 --seeds 72-81 --deadline $DL
$PY scripts/run_truck_scale_sweep.py final --config X1 --seeds 72-81 --workers 1 --deadline $DL
$PY scripts/run_truck_scale_sweep.py final --config X5 --seeds 72-81 --workers 1 --deadline $DL
