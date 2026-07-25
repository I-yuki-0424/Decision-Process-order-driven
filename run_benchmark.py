import sys
import os
import json
import traceback

sys.path.insert(0, os.path.abspath("."))

from src.pipeline.benchmark import run_hierarchical_benchmark_suite
from src.pipeline.plotter import plot_hierarchical_benchmark_results

def run(run_seq: str = "Run-Seq: #002"):
    try:
        print(f"=== Executing 5th-Idea Hierarchical Benchmark Suite (|A|=2000) [{run_seq}] ===")
        results = run_hierarchical_benchmark_suite(
            output_log_path="output/logs/execution_seq002.log",
            output_json_path="output/benchmark_hierarchical_seq002.json",
            run_seq=run_seq,
        )
        
        plot_hierarchical_benchmark_results([r._asdict() for r in results], run_seq=run_seq)
        print(f"Comparative plots generated successfully in output/plots/ [{run_seq}].")
        return results
    except Exception as e:
        print("ERROR IN BENCHMARK EXECUTION:")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    run()
