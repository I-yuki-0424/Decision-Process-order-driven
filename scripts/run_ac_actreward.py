"""Wrapper for Kaggle: train the learned action-reward model into <ar_dir> if absent, then run the actor-critic arms.
  python scripts/run_ac_actreward.py <ar_dir> <run_actor_critic_litmus.py args...>
"""
import os
import subprocess
import sys

ar_dir, rest = sys.argv[1], sys.argv[2:]
if not os.path.exists(os.path.join(ar_dir, "ar_full.pkl")):
    subprocess.run([sys.executable, "-u", "scripts/run_action_reward_model.py", "--out", ar_dir], check=True)
sys.exit(subprocess.run([sys.executable, "-u", "scripts/run_actor_critic_litmus.py", "--ar-dir", ar_dir] + rest).returncode)
