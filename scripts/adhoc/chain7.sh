#!/bin/bash
# waits for the X1 final (seeds 72-81), then the X5 final
PY=C:/v/e/Scripts/python.exe
DL=2026-10-08T07:00
while [ "$(ls output/phase1/truck_scale_t024/final/X1__s7[2-9].json output/phase1/truck_scale_t024/final/X1__s8[01].json 2>/dev/null | wc -l)" -lt 10 ] || [ -n "$(docker ps -q)" ]; do sleep 20; done
$PY scripts/run_truck_scale_sweep.py final --config X5 --seeds 72-81 --workers 1 --deadline $DL
