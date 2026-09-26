"""(Root cause of odd names: launching kaggle_run_experiment.py from Git Bash with an argument like /kaggle/working/exp
rewrites it to C:/Program Files/Git/kaggle/working/exp; launch with MSYS_NO_PATHCONV=1.)
Fetch Kaggle kernel output when file names are absolute (/kaggle/working/...), which the stock CLI
mis-resolves on Windows (os.path.join with an absolute name escapes the target dir).
  python scripts/kaggle_fetch_abs.py bfloat16/dpod-candidates-convergence output_remote/<run-id>"""
import os, sys, requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

kernel, out = sys.argv[1], sys.argv[2]
owner, slug = kernel.split("/")
a = KaggleApi(); a.authenticate()
token, n = None, 0
while True:
    with a.build_kaggle_client() as k:
        req = ApiListKernelSessionOutputRequest(); req.user_name = owner; req.kernel_slug = slug
        if token: req.page_token = token
        req.page_size = 100
        resp = k.kernels.kernels_api_client.list_kernel_session_output(req)
    for it in resp.files or []:
        name = it.file_name.replace("C:/Program Files/Git", "")  # Git-Bash MSYS path mangling of /kaggle/working
        if name.endswith(".pkl"):
            continue  # checkpoints are large; results/logs are enough
        dst = os.path.join(out, name.lstrip("/\\").replace("kaggle/working/", "", 1))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "wb").write(requests.get(it.url).content); n += 1
    if resp.log and not token:
        open(os.path.join(out, slug + ".log"), "w", encoding="utf-8").write(resp.log)
    token = resp.next_page_token
    if not token: break
print("fetched", n, "files ->", out)
