![Bayesian Neural Architecture Search banner](assets/banner.svg)

# Bayesian Neural Architecture Search

This project evaluates Bayesian optimization for neural architecture search on NASBench-201. It compares surrogate models with different assumptions about smoothness, structure, and sample efficiency.

## Research question

How does surrogate choice affect optimization quality when architecture evaluations are expensive?

The search pipeline supports:

- Gaussian process surrogates
- Random forest surrogates
- Multilayer perceptron surrogates
- Graph neural network surrogates
- Static and changing benchmark regimes

The evaluation focuses on regret, rank correlation, search trajectories, and computational cost.

## Method

Architectures are encoded from the NASBench-201 topology space. A surrogate predicts validation performance, an acquisition rule selects the next candidate, and the observed result updates the model. The dynamic environment can switch between CIFAR-10, CIFAR-100, and ImageNet16-120 while reusing one benchmark API instance.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements_NAS.txt
```

Place the NATS-Bench topology archive in the repository root, then run:

```bash
python main.py
```

## Repository layout

```text
BO.py                    Bayesian optimization loop
dynamic_env.py           Non-stationary benchmark wrapper
encoder.py               Architecture encodings
nasbench201_space.py     NASBench-201 adapter
surrogates/              Surrogate model implementations
config.py                Experiment configuration
tests.py                 Core checks
```

## Reproducibility

Experiment seeds are centralized in `seed.py`. The benchmark archive is intentionally excluded from Git because of its size and upstream distribution terms.

## Scope

This is an empirical research implementation, not a claim that one surrogate is universally best across search spaces or compute budgets.
