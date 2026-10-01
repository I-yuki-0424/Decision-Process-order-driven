"""Exit when every result file in a directory has the expected number of seeds (used as a background waiter)."""
import glob
import json
import sys
import time

d, n_files, n_seeds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
while True:
    done = 0
    for f in glob.glob(d + "/*.json"):
        try:
            done += len(json.load(open(f))["seeds"]) >= n_seeds
        except Exception:
            pass
    if done >= n_files:
        print("complete", done)
        break
    time.sleep(120)
