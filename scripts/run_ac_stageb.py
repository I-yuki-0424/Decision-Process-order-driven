"""Wrapper for Kaggle: (re)train the Stage-A passive world model into <wm_dir> if absent (default args of run_passive_wm_v2.py = the
original 2026-09-27 settings: random behaviour policy, 256 envs x 250 steps, 6000 train steps), then run the actor-critic Stage-B arms.
  python scripts/run_ac_stageb.py <wm_dir> <run_actor_critic_litmus.py args...>
"""
import os
import subprocess
import sys

wm_dir, rest = sys.argv[1], sys.argv[2:]
if not os.path.exists(os.path.join(wm_dir, "wm_mlp_full.pkl")):
    subprocess.run([sys.executable, "-u", "scripts/run_passive_wm_v2.py", "--out", wm_dir], check=True)
sys.exit(subprocess.run([sys.executable, "-u", "scripts/run_actor_critic_litmus.py", "--wm-dir", wm_dir] + rest).returncode)
