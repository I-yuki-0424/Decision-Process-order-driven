#!/bin/bash
PY=C:/v/e/Scripts/python.exe
DL=2026-10-07T12:40
$PY scripts/run_baseline_warmup_check.py tune --seeds 2000-2002 --deadline $DL
