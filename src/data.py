import numpy as np

def generate_benchmark_data(n_trials=150, duration=0.3, dt=0.1, seed=42):
    """Generates standardized benchmark input trials for multi-seed robustness testing."""
    np.random.seed(seed)
    n_steps = int(duration / (dt / 1000.0))
    
    labels = np.random.choice([0, 1], size=n_trials)
    trials = []
    for label in labels:
        base = 0.7 if label == 1 else -0.7
        signal = base * np.ones(n_steps) + np.random.normal(0, 0.3, size=n_steps)
        trials.append(signal)
        
    return labels, np.array(trials), dt