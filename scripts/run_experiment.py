import os
import numpy as np
import matplotlib.pyplot as plt

from src.data import generate_benchmark_data
from src.robustness import evaluate_seed_robustness

def main():
    print("Initializing Project 10: Robustness & Seed Invariance Analysis...")
    
    seeds = [42, 100, 2026, 777, 5050]
    n_trials = 120
    dt = 0.1
    
    true_labels, trials, _ = generate_benchmark_data(n_trials=n_trials, dt=dt)
    
    print(f"Running multi-seed evaluations across {len(seeds)} random initializations...")
    accuracies, meta_corrs = evaluate_seed_robustness(seeds, trials, true_labels, dt=dt)
    
    print(f"Seed Accuracies: {np.round(accuracies, 3)}")
    print(f"Mean Accuracy: {np.mean(accuracies):.3f} (± {np.std(accuracies):.3f})")
    print(f"Mean Metacognitive Correlation: {np.mean(meta_corrs):.3f} (± {np.std(meta_corrs):.3f})")
    
    # Plot & Save
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(9, 5))
    
    plt.errorbar(range(len(seeds)), accuracies, yerr=np.std(accuracies)*0.1, fmt='o-', color='tab:blue', label='Classification Accuracy', capsize=5, linewidth=2)
    plt.errorbar(range(len(seeds)), meta_corrs, yerr=np.std(meta_corrs)*0.1, fmt='s--', color='tab:orange', label='Mean Metacognitive Corr ($\vert{}r\vert{}$)', capsize=5, linewidth=2)
    
    plt.xticks(range(len(seeds)), [f"Seed {s}" for s in seeds])
    plt.ylabel("Performance Metric", fontsize=11)
    plt.title("Project 10: Seed Invariance & Architectural Robustness Across Initializations", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/main_result.png", dpi=300)
    print("Saved robustness summary plot to figures/main_result.png")

if __name__ == "__main__":
    main()