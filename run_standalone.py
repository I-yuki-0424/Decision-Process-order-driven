import sys
import os
import json
import traceback

sys.path.insert(0, os.path.abspath("."))

from src.pipeline.benchmark import run_offpolicy_abstraction_benchmark_suite
from src.pipeline.plotter import plot_offpolicy_benchmark_results

def main(run_seq: str = "Run-Seq: #003"):
    try:
        print(f"Starting 5th-Idea Off-Policy & Abstraction Embedding Benchmark Suite (|A|=2000) [{run_seq}]...")
        results, loss_hist = run_offpolicy_abstraction_benchmark_suite(
            output_log_path="output/logs/execution_seq003.log",
            output_json_path="output/benchmark_offpolicy_seq003.json",
            run_seq=run_seq,
        )
        print("Suite completed! Results count:", len(results))
        
        plot_offpolicy_benchmark_results([r._asdict() for r in results], loss_hist, run_seq=run_seq)
        print("Plotting completed successfully!")
        return results
    except Exception as e:
        print("EXCEPTION ENCOUNTERED:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
