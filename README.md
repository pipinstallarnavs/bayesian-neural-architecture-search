# Neural-Architecture-Search-Project

# Bayesian Optimization for Neural Architecture Search with Different Surrogate Models

## 📌 Project Overview
This repository explores **Bayesian Optimization (BO)** for **Neural Architecture Search (NAS)**, comparing multiple surrogate models:
- **Gaussian Process (GP)**
- **Random Forest (RF)**
- **Multi-Layer Perceptron (MLP)**
- **Graph Neural Network (GNN)**

The goal is to evaluate trade-offs between **accuracy, scalability, and sample efficiency** of these surrogate models when used inside a BO-driven NAS pipeline.

## 🏗️ Motivation
Traditional NAS methods are computationally expensive. Surrogate-assisted Bayesian Optimization provides a more **efficient search strategy**, but performance depends heavily on the choice of surrogate.  
This project investigates:
- Which surrogate models perform best across different benchmarks (NASBench-201, possibly NASBench-301/DARTS).  
- The trade-offs between **expressivity vs computational overhead**. We will try to compare compute hours and accuracy
- Metrics like **rank correlation, regret, and trajectory visualizations**.

## ⚙️ Methodology
1. Implement a baseline BO-NAS pipeline.  
2. Swap surrogate models (GP, RF, MLP, GNN).  
3. Evaluate across benchmarks.  
4. Compare optimization efficiency and architecture quality.  

## 📊 Expected Contributions
- **Empirical comparison** of surrogate models in BO-NAS.  
- **Insights** into scalability of GNN surrogates.  
- **Benchmark results** across datasets.  
- **Reproducibility**: clean, modular implementation.


