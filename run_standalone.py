import sys
import os
import json
import traceback

sys.path.insert(0, os.path.abspath("."))

from src.pipeline.craftax_benchmark import run_craftax_benchmark_suite
from src.pipeline.plotter import plot_craftax_benchmark_results

def main(run_seq: str = "Run-Seq: #004"):
    try:
        print(f"Starting Craftax-Classic Reinforcement Learning Benchmark Suite [{run_seq}]...")
        results = run_craftax_benchmark_suite(
            output_log_path="output/logs/execution_seq004.log",
            output_json_path="output/benchmark_craftax_seq004.json",
            run_seq=run_seq,
        )
        print("Suite completed! Results count:", len(results))
        
        plot_craftax_benchmark_results([r._asdict() for r in results], run_seq=run_seq)
        print("Plotting completed successfully!")
        return results
    except Exception as e:
        print("EXCEPTION ENCOUNTERED:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
