import sys
import os
import json
import traceback

sys.path.insert(0, os.path.abspath("."))

from src.pipeline.benchmark import run_full_benchmark_suite
from src.pipeline.plotter import plot_full_benchmark_results

def run():
    try:
        print("=== Executing Gymnax Decision Transformer Benchmark Suite ===")
        results = run_full_benchmark_suite()
        print(f"Benchmark completed successfully! Evaluated {len(results)} variants.")
        
        with open("output/benchmark_results.json", "r", encoding="utf-8") as f:
            res_data = json.load(f)
        
        plot_full_benchmark_results(res_data)
        print("Comparative plots generated in output/plots/.")
        return results
    except Exception as e:
        print("ERROR IN BENCHMARK EXECUTION:")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    run()
