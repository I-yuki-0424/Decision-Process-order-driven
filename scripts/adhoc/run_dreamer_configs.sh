#!/bin/bash
# Dreamer-lite fidelity study: same task/data, rssm arm only vs hn reference, increasing DreamerV3 fidelity.
O=output/experiments/2026-09-26_rerun/dreamer
C="--tasks kepler_pert pendulum --sizes 512 --noises 0.02 --seeds 0 1 2"
python scripts/run_hybrid_worldmodel.py --out $O/big       --arms rssm    $C --steps 3000 --dh 128 --hid 128
python scripts/run_hybrid_worldmodel.py --out $O/v3        --arms rssm    $C --steps 3000 --dh 128 --hid 128 --v3
python scripts/run_hybrid_worldmodel.py --out $O/v3_16x16  --arms rssm    $C --steps 3000 --dh 128 --hid 128 --v3 --ng 16 --nc 16
python scripts/run_hybrid_worldmodel.py --out $O/v3_long   --arms rssm    $C --steps 12000 --dh 128 --hid 128 --v3 --ng 16 --nc 16
