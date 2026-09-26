"""Self-contained job (for Kaggle): Stage A (data + WM) then Stage B arms. Usage: run_pwm_job.py <arms> <seeds> [more B args]."""
import subprocess, sys
arms, seeds = sys.argv[1].split(","), sys.argv[2].split(",")
extra = sys.argv[3:]
py = sys.executable
subprocess.run([py, "-u", "scripts/run_passive_wm_v2.py", "--out", "/kaggle/working/wm", "--envs", "384"], check=True)
subprocess.run([py, "-u", "scripts/run_passive_policy_v2.py", "--out", "/kaggle/working/exp", "--wm-dir", "/kaggle/working/wm",
                "--arms", *arms, "--seeds", *seeds, *extra], check=True)
