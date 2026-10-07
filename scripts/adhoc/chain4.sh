#!/bin/bash
PY=C:/v/e/Scripts/python.exe
DL=2026-10-07T12:40
$PY scripts/run_baseline_warmup_check.py final --configs G4 --deadline $DL
$PY scripts/run_truck_scale_sweep.py final --config S2 --workers 1 --deadline $DL
$PY scripts/run_truck_scale_sweep.py final --config S3 --workers 1 --deadline $DL
