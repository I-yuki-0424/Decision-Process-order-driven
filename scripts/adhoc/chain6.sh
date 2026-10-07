#!/bin/bash
# waits for the running baseline G4 final (seeds 72-81), then recipe round R, then the X1 final
PY=C:/v/e/Scripts/python.exe
DL=2026-10-08T07:00
while [ "$(ls output/phase1/truck_scale_t024/final/G4__s7[2-9].json output/phase1/truck_scale_t024/final/G4__s81.json 2>/dev/null | wc -l)" -lt 9 ] || [ -n "$(docker ps -q)" ]; do sleep 20; done
$PY scripts/run_truck_scale_sweep.py run --stage R --configs R1,R2,R3,R4,R5,R6 --seeds 2000-2003 --workers 1 --deadline $DL
$PY scripts/run_truck_scale_sweep.py final --config X1 --seeds 72-81 --workers 1 --deadline $DL
