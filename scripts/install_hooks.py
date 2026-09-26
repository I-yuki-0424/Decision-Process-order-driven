"""One-time per clone: point git at the tracked .githooks/ (LFS hooks + requirements sync)."""
import os
import subprocess

subprocess.run(["git", "config", "core.hooksPath", ".githooks"], check=True)
for name in os.listdir(".githooks"):
    try:
        os.chmod(os.path.join(".githooks", name), 0o755)
    except OSError:
        pass
print("core.hooksPath -> .githooks")
