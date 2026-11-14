# Robustness & Seed Invariance (`lsm-10-confidence-robustness`)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/)
[![Brian2](https://img.shields.io/badge/Simulator-Brian2-orange.svg)](https://brian2.readthedocs.io/)

> *Evaluating topological robustness, seed invariance, and statistical stability of spiking neural reservoir performance and metacognitive confidence metrics across multiple random initializations.*

---

## Abstract & Scientific Rationale

![Main Result](figures/main_result.png)

As the capstone of our 10-project Liquid State Machine sprint, **Project 10** addresses a critical question in neuromorphic engineering: *Are emergent reservoir properties—such as classification accuracy and metacognitive confidence-encoding—dependent on specific random weight initializations, or are they robust, invariant features of recurrent spiking networks?*

By systematically instantiating our 200-neuron Leaky Integrate-and-Fire (LIF) network across multiple random seeds, this project measures the variance and statistical stability of both primary decision decoding and secondary uncertainty monitoring.

---

## Repository Architecture

```text
lsm-10-confidence-robustness/
├── data/
│   └── README.md
├── figures/
│   └── main_result.png
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── reservoir.py
│   └── robustness.py
├── scripts/
│   └── run_experiment.py
├── .gitignore
├── requirements.txt
└── README.md