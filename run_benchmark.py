import sys
import os
import json
import traceback

sys.path.insert(0, os.path.abspath("."))

from src.pipeline.craftax_benchmark import run_craftax_benchmark_suite
from src.pipeline.plotter import plot_craftax_benchmark_results

def run(run_seq: str = "Run-Seq: #005"):
    try:
        print(f"=== Executing Scaled 1,000-Episode Craftax-Classic RL Benchmark Suite [{run_seq}] ===")
        results = run_craftax_benchmark_suite(
            output_log_path="output/logs/execution_seq005.log",
            output_json_path="output/benchmark_craftax_seq005.json",
            run_seq=run_seq,
            train_episodes=1000,
            eval_episodes=50,
        )
        
        plot_craftax_benchmark_results([r._asdict() for r in results], run_seq=run_seq)
        print(f"Comparative Craftax 1,000-ep plots generated successfully in output/plots/ [{run_seq}].")
        return results
    except Exception as e:
        print("ERROR IN CRAFTAX BENCHMARK EXECUTION:")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    run()
