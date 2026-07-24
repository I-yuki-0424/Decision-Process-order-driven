import sys
import os
import json
import traceback

sys.path.insert(0, os.path.abspath("."))

from src.pipeline.benchmark import run_full_benchmark_suite
from src.pipeline.plotter import plot_full_benchmark_results

if __name__ == "__main__":
    try:
        print("Starting Benchmark Suite...")
        results = run_full_benchmark_suite()
        print("Suite completed! Results count:", len(results))
        
        with open("output/benchmark_results.json", "r", encoding="utf-8") as f:
            res_data = json.load(f)
        
        plot_full_benchmark_results(res_data)
        print("Plotting completed successfully!")
    except Exception as e:
        print("EXCEPTION ENCOUNTERED:")
        traceback.print_exc()
