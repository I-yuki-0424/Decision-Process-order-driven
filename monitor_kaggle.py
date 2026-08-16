import subprocess
import time
import sys
import os

KERNEL = "bfloat16/craftax-classic-1000-episode-rl-benchmark"

def check_status():
    result = subprocess.run(["kaggle", "kernels", "status", KERNEL], capture_output=True, text=True)
    return result.stdout.lower()

def main():
    print(f"Monitoring Kaggle kernel: {KERNEL}...")
    
    while True:
        status = check_status()
        if "error" in status:
            print("Kernel failed with ERROR status!")
            break
        elif "complete" in status:
            print("Kernel finished SUCCESSFULLY!")
            break
        elif "running" in status:
            pass # Keep waiting
        elif "queued" in status:
            pass # Keep waiting
        else:
            print(f"Current status: {status.strip()}")
            
        time.sleep(15)
        
    print("Fetching outputs...")
    os.makedirs("kaggle_logs", exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", KERNEL, "-p", "kaggle_logs", "--force"])
    print("Monitor complete. Reading log:")
    
    log_path = "kaggle_logs/craftax-classic-1000-episode-rl-benchmark.log"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            lines = f.readlines()
            print("".join(lines[-50:]))
    else:
        print("Log file not found.")

if __name__ == "__main__":
    main()
