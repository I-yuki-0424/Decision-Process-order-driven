import sys
import os
import json
import traceback

sys.path.insert(0, os.path.abspath("."))

from src.pipeline.benchmark import run_layer_depth_scaling_experiment
from src.pipeline.plotter import plot_layer_depth_scaling_and_bottlenecks, plot_full_benchmark_results

def run(run_seq: str = "Run-Seq: #001"):
    try:
        print(f"=== Executing Layer Depth Scaling & MDP Bottleneck Suite [{run_seq}] ===")
        scaling_results, friction_data = run_layer_depth_scaling_experiment(
            layer_list=[2, 4, 8, 12],
            d_model=512,
            max_steps=100,
            run_seq=run_seq,
        )
        
        plot_layer_depth_scaling_and_bottlenecks(scaling_results, friction_data, run_seq=run_seq)
        plot_full_benchmark_results(scaling_results, run_seq=run_seq)
        print(f"Comparative plots generated successfully in output/plots/ [{run_seq}].")
        return scaling_results, friction_data
    except Exception as e:
        print("ERROR IN BENCHMARK EXECUTION:")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    run()
