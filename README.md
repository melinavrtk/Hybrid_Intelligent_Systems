# Hybrid Intelligent Systems

This repository contains implementations of Hybrid Intelligent Systems, demonstrating the integration of **Neural Networks, Fuzzy Logic, and Evolutionary/Probabilistic Algorithms**. The projects focus on solving complex control, optimization, and non-linear classification problems by combining classical mathematical approaches with modern computational intelligence.

## 📂 Repository Structure

### 1. Fuzzy Logic Control (`01_fuzzy_lighting_controller.py`)
* Implements a **Mamdani Fuzzy Inference System** from scratch using `scikit-fuzzy`.
* Dynamically controls a lighting system based on ambient brightness and its rate of change.
* Features custom Membership Functions (Trapezoidal, Triangular, Sigmoid), fuzzy set intersections, rule aggregations, and Centroid defuzzification.

### 2. Hybrid Regression Optimizers (`02_hybrid_regression_optimizers.py`)
* Solves a Polynomial Regression problem (Boston Housing dataset) by benchmarking four fundamentally different optimization paradigms:
  1. **Closed-Form Solutions:** Ordinary Least Squares (OLS) and Ridge Regression.
  2. **Gradient-Based Learning:** Implementation of the **Adam Optimizer** from scratch using `numpy`.
  3. **Evolutionary Algorithms:** Genetic Algorithms (Differential Evolution) via `scipy.optimize`.
  4. **Probabilistic Optimization:** Simulated Annealing (Dual Annealing).
* Compares convergence, test MSE, and computational efficiency across all methods.

### 3. Neuro-Fuzzy Systems & Deep Learning (`03_neuro_fuzzy_xor_anfis.py`)
* Tackles the non-linear 3-bit XOR problem using two distinct hybrid approaches:
  1. **Custom Adam MLP:** A Multi-Layer Perceptron built entirely from scratch in `numpy`, featuring exact mathematical implementations of forward propagation, backpropagation, and Adam weight updates.
  2. **ANFIS Equivalent:** An Adaptive Neuro-Fuzzy Inference System equivalent, built with **TensorFlow/Keras**, modeling Gaussian membership functions and 0th-order Sugeno outputs via a specialized neural architecture.

## 🛠️ Skills & Technologies Highlighted
* **Computational Intelligence:** Fuzzy Logic, Genetic Algorithms, Simulated Annealing.
* **Deep Learning Mathematics:** Building forward/backward propagation and advanced optimizers (Adam) purely with linear algebra.
* **Frameworks & Libraries:** `Python`, `numpy`, `scikit-fuzzy`, `scipy.optimize`, `TensorFlow`, `Keras`, `matplotlib`.

## 📌 Acknowledgments & Context
The mathematical foundations and initial implementations for these projects were developed as part of my undergraduate coursework at the **University of West Attica (Biomedical Engineering)**. 

The current repository represents a modernized evolution of those academic assignments. The original MATLAB/procedural scripts have been translated, rewritten into Object-Oriented Python, vectorized for performance, and structured into professional pipelines to bridge the gap between academic theory and industry standards.

---
*Curated, refactored, and optimized by a final-year Biomedical Engineering student (University of West Attica), specializing in AI and Medical Data Science.*
