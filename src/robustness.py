import numpy as np
from sklearn.linear_model import LogisticRegression

def evaluate_seed_robustness(seeds, trials, true_labels, dt=0.1):
    """Executes multi-seed reservoir simulations to quantify stability of decoding accuracy and meta-correlation."""
    accuracies = []
    meta_corrs = []
    
    n_trials = len(true_labels)
    n_neurons = 200
    duration = len(trials[0]) * (dt / 1000.0)
    
    for seed in seeds:
        from brian2 import run, second, ms
        neurons, synapses, spike_mon = build_seeded_reservoir(n_neurons=n_neurons, dt=dt, seed=seed)
        
        total_dur = len(trials.flatten()) * (dt / 1000.0)
        run(total_dur * second)
        
        # Extract features
        spike_times = spike_mon.t / ms
        spike_indices = spike_mon.i
        steps_per_trial = int(duration / (dt / 1000.0))
        
        X = np.zeros((n_trials, n_neurons))
        for i in range(n_trials):
            t_start = i * (steps_per_trial * dt)
            t_end = (i + 1) * (steps_per_trial * dt)
            mask = (spike_times >= t_start) & (spike_times < t_end)
            if np.sum(mask) > 0:
                counts, _ = np.histogram(spike_indices[mask], bins=np.arange(n_neurons + 1))
                X[i] = counts / duration
                
        # Model evaluation
        model = LogisticRegression(C=1.0).fit(X, true_labels)
        acc = model.score(X, true_labels)
        preds = model.predict(X)
        correctness = (preds == true_labels).astype(int)
        
        # Mean absolute confidence-correlation across active neurons
        corrs = []
        for n_idx in range(n_neurons):
            r = X[:, n_idx]
            if np.std(r) > 0:
                c = np.corrcoef(r, correctness)[0, 1]
                if not np.isnan(c):
                    corrs.append(np.abs(c))
        mean_meta = np.mean(corrs) if corrs else 0.0
        
        accuracies.append(acc)
        meta_corrs.append(mean_meta)
        
    return np.array(accuracies), np.array(meta_corrs)