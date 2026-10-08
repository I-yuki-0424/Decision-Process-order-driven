"""TASK-20261008-025: run a batch of scripts/run_epa_mini.py jobs (from run_t025_sweep.py configs) on a Kaggle GPU kernel.

  python scripts/kaggle_t025.py build --kernel t025-a --stage K --configs E1,K1 --seeds 3004-3005      # writes kaggle_kernel_t025/<kernel>/
  python scripts/kaggle_t025.py push --kernel t025-a
  python scripts/kaggle_t025.py status --kernel t025-a
  python scripts/kaggle_t025.py fetch --kernel t025-a        # result JSONs -> output/phase1/push_t025/<stage>/ (+ logs)

The notebook embeds src/ and scripts/run_epa_mini.py (base64), installs craftax, and runs one job per visible GPU in parallel
(CUDA_VISIBLE_DEVICES), the same command lines as the local Docker driver. Package versions are recorded per result file
(run_provenance), so Kaggle runs are distinguishable from local ones. Tuning only (role=tune) unless --final is given.
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_t025_sweep as sw  # noqa: E402
import run_phase1_recipe_sweep as rs  # noqa: E402

KROOT = "kaggle_kernel_t025"
USER = "bfloat16"

RUNNER = r'''
import json, os, subprocess, sys, threading, time, queue
T0 = time.time()
jobs = json.load(open("jobs.json"))
ngpu = len([l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines() if l.startswith("GPU")])
print("GPUs:", ngpu, "jobs:", len(jobs), flush=True)
os.makedirs("/kaggle/working/logs", exist_ok=True)
q = queue.Queue()
for j in jobs:
    q.put(j)
LIMIT = float(os.environ.get("T025_TIME_LIMIT", str(11 * 3600)))
def worker(g):
    while True:
        try:
            j = q.get_nowait()
        except queue.Empty:
            return
        if time.time() - T0 + j["sec"] * 1.5 > LIMIT:
            print("time limit: skip", j["name"], flush=True)
            continue
        out = "/kaggle/working/" + j["out"]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cmd = list(j["cmd"])
        cmd[cmd.index("--out") + 1] = out
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1", **j["env"])
        t = time.time()
        with open("/kaggle/working/logs/" + j["name"] + ".log", "w") as lf:
            rc = subprocess.run([sys.executable if c == "python" else c for c in cmd], stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
        print(f"gpu{g} {j['name']}: rc={rc} {time.time() - t:.0f}s", flush=True)
ths = [threading.Thread(target=worker, args=(g,)) for g in range(max(ngpu, 1))]
for i, t in enumerate(ths):
    t.start()
    time.sleep(20)
for t in ths:
    t.join()
print("ALL DONE", time.time() - T0, flush=True)
'''


def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def build(a):
    prov = rs.host_provenance()
    if a.final:
        jobs = sw.jobs_for(a.stage, a.configs.split(","), sw.parse_seeds(a.seeds), role="final", offset=sw.TEST_SEED_OFFSET, protocol="EP-A",
                           budget=f"TASK-025: configs fixed from screening on tuning seeds 3000-3999 ({a.configs})")
        if prov["GIT_DIRTY"] == "1":
            raise SystemExit("final jobs need a clean tree")
    else:
        seeds = sw.parse_seeds(a.seeds)
        if min(seeds) < 3000:
            raise SystemExit("tuning seeds are 3000-3999")
        jobs = sw.jobs_for(a.stage, a.configs.split(","), seeds)
        for spec in a.extra:   # additional "stage:configs:seeds" groups in the same kernel
            st, cf, sd = spec.split(":")
            jobs += sw.jobs_for(st, cf.split(","), sw.parse_seeds(sd))
    for j in jobs:
        j["env"] = {**prov, "KAGGLE_RUN": a.kernel}
    kdir = os.path.join(KROOT, a.kernel)
    os.makedirs(kdir, exist_ok=True)
    files = [os.path.join(r, f).replace("\\", "/") for r, _, fs in os.walk("src") for f in fs if f.endswith(".py")]
    files += ["scripts/run_epa_mini.py"]
    dec = ["import base64, os, json\n"]
    for f in sorted(files):
        dec.append(f"os.makedirs({os.path.dirname(f)!r}, exist_ok=True); open({f!r}, 'wb').write(base64.b64decode({b64(f)!r}))\n")
    dec.append(f"open('jobs.json', 'wb').write(base64.b64decode({base64.b64encode(json.dumps(jobs).encode()).decode()!r}))\n")
    install = ["import subprocess, sys\n",
               "r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'craftax==1.6.1', 'optax', 'flax', 'gymnax'], capture_output=True, text=True)\n",
               "print(r.stdout[-1500:], r.stderr[-3000:])\n",
               "r = subprocess.run([sys.executable, '-c', 'import jax, craftax; print(jax.__version__, jax.devices())'], capture_output=True, text=True)\n",
               "print(r.stdout, r.stderr[-3000:])\n"]
    cells = [install, dec, [l + "\n" for l in RUNNER.splitlines()]]
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c} for c in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(os.path.join(kdir, "run.ipynb"), "w"))
    meta = {"id": f"{USER}/dpod-{a.kernel}", "title": f"DPOD {a.kernel}", "code_file": "run.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    json.dump(meta, open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=1)
    json.dump(jobs, open(os.path.join(kdir, "jobs.json"), "w"), indent=1)
    print(f"built {kdir}: {len(jobs)} jobs, commit {prov['GIT_COMMIT'][:8]} dirty={prov['GIT_DIRTY']}")


def kaggle(*args, check=True):
    env = dict(os.environ, KAGGLE_CONFIG_DIR=os.getcwd())
    exe = os.path.join(os.path.dirname(sys.executable), "kaggle.exe" if os.name == "nt" else "kaggle")
    r = subprocess.run([exe, *args], capture_output=True, text=True, env=env)
    if check and r.returncode:
        raise SystemExit(r.stdout + r.stderr)
    return r.stdout + r.stderr


def fetch(a):
    kdir = os.path.join(KROOT, a.kernel)
    tmp = os.path.join(kdir, "out")
    shutil.rmtree(tmp, ignore_errors=True)
    print(kaggle("kernels", "output", f"{USER}/dpod-{a.kernel}", "-p", tmp, "--force", check=False)[-500:])
    jobs = json.load(open(os.path.join(kdir, "jobs.json")))
    n = 0
    for j in jobs:
        src = os.path.join(tmp, j["out"])
        if os.path.exists(src):
            os.makedirs(os.path.dirname(j["out"]), exist_ok=True)
            shutil.copy(src, j["out"])
            n += 1
        lg = os.path.join(tmp, "logs", j["name"] + ".log")
        if os.path.exists(lg):
            d = os.path.join(os.path.dirname(j["out"]), "logs")
            os.makedirs(d, exist_ok=True)
            shutil.copy(lg, os.path.join(d, j["name"] + ".kaggle.log"))
    print(f"fetched {n}/{len(jobs)} result files")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "push", "status", "fetch"])
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--stage", default="K")
    ap.add_argument("--configs", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--extra", action="append", default=[], help="more job groups 'stage:configs:seeds' (tuning only)")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a)
    elif a.cmd == "push":
        print(kaggle("kernels", "push", "-p", os.path.join(KROOT, a.kernel)))
    elif a.cmd == "status":
        print(kaggle("kernels", "status", f"{USER}/dpod-{a.kernel}", check=False))
    else:
        fetch(a)


if __name__ == "__main__":
    main()
