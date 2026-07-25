import sys
import os
import json
import traceback

sys.path.insert(0, os.path.abspath("."))

from src.pipeline.benchmark import run_layer_depth_scaling_experiment
from src.pipeline.plotter import plot_layer_depth_scaling_and_bottlenecks, plot_full_benchmark_results

def main(run_seq: str = "Run-Seq: #001"):
    try:
        print(f"Starting Layer Depth Scaling & Bottleneck Suite [{run_seq}]...")
        scaling_results, friction_data = run_layer_depth_scaling_experiment(
            layer_list=[2, 4, 8, 12],
            d_model=512,
            max_steps=100,
            run_seq=run_seq,
        )
        print("Suite completed! Layer variants count:", len(scaling_results))
        
        plot_layer_depth_scaling_and_bottlenecks(scaling_results, friction_data, run_seq=run_seq)
        plot_full_benchmark_results(scaling_results, run_seq=run_seq)
        print("Plotting completed successfully!")
        return scaling_results, friction_data
    except Exception as e:
        print("EXCEPTION ENCOUNTERED:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
