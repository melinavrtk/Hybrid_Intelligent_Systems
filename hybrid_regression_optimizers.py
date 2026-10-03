# =========================================================
# Hybrid Intelligent Systems
# Assignment 2: Polynomial Regression Optimization Suite
# Methods: Closed-Form, Gradient Descent, Adam, GA, Simulated Annealing
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import differential_evolution, dual_annealing

class HybridRegressionOptimizer:
    """
    A comprehensive suite solving Polynomial Regression using 4 distinct paradigms:
    1. Closed-Form (Pseudo-inverse & Ridge)
    2. Gradient-Based (Batch GD, Mini-Batch SGD, Adam)
    3. Evolutionary (Genetic Algorithm)
    4. Probabilistic (Simulated Annealing)
    """

    def __init__(self, degree=5):
        self.degree = degree
        self.X_train, self.X_test = None, None
        self.Y_train, self.Y_test = None, None
        self.X_train_poly, self.X_test_poly = None, None
        self.mean_x, self.std_x = None, None

    def load_and_preprocess(self, csv_path='BostonHousing.csv'):
        """ Loads data, extracts the 6th feature (Rooms), normalizes, and builds polynomials. """
        try:
            # Attempt to load local CSV
            data = pd.read_csv(csv_path)
            data_matrix = data.values
        except FileNotFoundError:
            # Fallback for out-of-the-box execution if CSV is missing
            print("Local CSV not found. Fetching Boston Housing from OpenML...")
            from sklearn.datasets import fetch_openml
            boston = fetch_openml('boston', version=1, as_frame=True, parser='auto')
            data_matrix = boston.frame.values
            
        # Target is the last column, Feature is the 6th column (RM - Number of Rooms)
        X_raw = data_matrix[:, 5].astype(float)
        Y_raw = data_matrix[:, -1].astype(float)
        
        # Train-Test Split (70-30)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X_raw, Y_raw, test_size=0.3, random_state=42
        )
        
        # Normalization (Z-score) based on training data
        self.mean_x = np.mean(self.X_train)
        self.std_x = np.std(self.X_train, ddof=1)
        
        X_train_norm = (self.X_train - self.mean_x) / self.std_x
        X_test_norm = (self.X_test - self.mean_x) / self.std_x
        
        # Build Polynomial Features Matrix [1, x, x^2, ..., x^K]
        self.X_train_poly = self._build_polynomial(X_train_norm)
        self.X_test_poly = self._build_polynomial(X_test_norm)
        print(f"Data Prepared: Training Samples={len(self.Y_train)}, Polynomial Degree={self.degree}")

    def _build_polynomial(self, X):
        """ Constructs the polynomial design matrix A. """
        A = np.ones((len(X), 1))
        for k in range(1, self.degree + 1):
            A = np.column_stack((A, X**k))
        return A

    @staticmethod
    def mse_cost(theta, X, Y):
        """ Mean Squared Error Cost Function. """
        predictions = X @ theta
        return np.mean((predictions - Y)**2)

    @staticmethod
    def mse_gradient(theta, X, Y):
        """ Computes the exact mathematical gradient of the MSE. """
        m = len(Y)
        predictions = X @ theta
        return (2 / m) * (X.T @ (predictions - Y))

    # =========================================================
    # 1. CLOSED-FORM SOLUTIONS
    # =========================================================
    def closed_form_solutions(self, lambda_ridge=0.1):
        """ Exact mathematical solutions using the Normal Equation and Ridge penalty. """
        print("\n--- 1. Closed-Form Solutions ---")
        
        # Standard OLS: theta = (X^T * X)^-1 * X^T * Y
        theta_ols = np.linalg.pinv(self.X_train_poly.T @ self.X_train_poly) @ (self.X_train_poly.T @ self.Y_train)
        mse_ols = self.mse_cost(theta_ols, self.X_test_poly, self.Y_test)
        print(f"OLS Test MSE (K={self.degree}): {mse_ols:.4f}")
        
        # Ridge Regression
        I = np.eye(self.degree + 1)
        I[0, 0] = 0 # Do not penalize the bias term
        theta_ridge = np.linalg.pinv(self.X_train_poly.T @ self.X_train_poly + lambda_ridge * I) @ (self.X_train_poly.T @ self.Y_train)
        mse_ridge = self.mse_cost(theta_ridge, self.X_test_poly, self.Y_test)
        print(f"Ridge Test MSE (\u03BB={lambda_ridge}): {mse_ridge:.4f}")
        
        return theta_ols

    # =========================================================
    # 2. GRADIENT-BASED OPTIMIZERS
    # =========================================================
    def adam_optimizer(self, learning_rate=0.1, max_iter=1000, tol=1e-6):
        """ Implements the Adam Optimizer from scratch using numpy. """
        print("\n--- 2. Adam Optimizer (Gradient-Based) ---")
        n_params = self.degree + 1
        theta = np.zeros(n_params)
        
        # Adam Hyperparameters
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        m, v = np.zeros(n_params), np.zeros(n_params)
        
        loss_history = []
        
        for t in range(1, max_iter + 1):
            grad = self.mse_gradient(theta, self.X_train_poly, self.Y_train)
            loss = self.mse_cost(theta, self.X_train_poly, self.Y_train)
            loss_history.append(loss)
            
            if np.linalg.norm(grad) < tol:
                break
                
            # Momentum & RMSprop updates
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            
            # Bias corrections
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            # Parameter update
            theta = theta - learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon))
            
        test_mse = self.mse_cost(theta, self.X_test_poly, self.Y_test)
        print(f"Adam Optimizer Converged in {t} iterations.")
        print(f"Adam Test MSE: {test_mse:.4f}")
        return theta, loss_history

    # =========================================================
    # 3. EVOLUTIONARY & PROBABILISTIC OPTIMIZERS
    # =========================================================
    def genetic_algorithm(self):
        """ Utilizes Differential Evolution (GA) to find optimal polynomial weights. """
        print("\n--- 3. Genetic Algorithm (Evolutionary) ---")
        bounds = [(-20, 20)] * (self.degree + 1)
        
        result = differential_evolution(
            self.mse_cost, 
            bounds=bounds, 
            args=(self.X_train_poly, self.Y_train),
            maxiter=50, 
            popsize=15,
            disp=False
        )
        
        test_mse = self.mse_cost(result.x, self.X_test_poly, self.Y_test)
        print(f"GA Optimization Successful: {result.success}")
        print(f"GA Test MSE: {test_mse:.4f}")
        return result.x

    def simulated_annealing(self):
        """ Utilizes Dual Annealing (SA) to escape local minima in weight space. """
        print("\n--- 4. Simulated Annealing (Probabilistic) ---")
        bounds = [(-20, 20)] * (self.degree + 1)
        
        result = dual_annealing(
            self.mse_cost, 
            bounds=bounds, 
            args=(self.X_train_poly, self.Y_train),
            maxiter=500
        )
        
        test_mse = self.mse_cost(result.x, self.X_test_poly, self.Y_test)
        print(f"SA Optimization Successful: {result.success}")
        print(f"SA Test MSE: {test_mse:.4f}")
        return result.x

    # =========================================================
    # VISUALIZATION
    # =========================================================
    def plot_results(self, theta_dict):
        """ Plots the Regression Curves for all optimization methods. """
        plt.figure(figsize=(10, 6))
        
        # Plot original test data
        plt.scatter(self.X_test, self.Y_test, color='gray', alpha=0.5, label='Actual Data')
        
        # Generate smooth points for plotting curves
        X_smooth = np.linspace(min(self.X_test), max(self.X_test), 200)
        X_smooth_norm = (X_smooth - self.mean_x) / self.std_x
        X_smooth_poly = self._build_polynomial(X_smooth_norm)
        
        colors = ['b', 'r', 'g', 'm']
        for i, (method_name, theta) in enumerate(theta_dict.items()):
            predictions = X_smooth_poly @ theta
            plt.plot(X_smooth, predictions, color=colors[i % len(colors)], linewidth=2, label=method_name)
            
        plt.title(f'Hybrid Regressors Comparison (Polynomial Degree = {self.degree})')
        plt.xlabel('Number of Rooms (Normalized/Rescaled)')
        plt.ylabel('House Price')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    # Initialize Optimizer Suite for Polynomial K=5
    hybrid_suite = HybridRegressionOptimizer(degree=5)
    
    # Load and Preprocess Dataset
    hybrid_suite.load_and_preprocess()
    
    # Run all 4 distinct algorithmic paradigms
    theta_closed = hybrid_suite.closed_form_solutions()
    theta_adam, adam_loss = hybrid_suite.adam_optimizer(learning_rate=0.2)
    theta_ga = hybrid_suite.genetic_algorithm()
    theta_sa = hybrid_suite.simulated_annealing()
    
    # Collect results and plot comparison
    results_dict = {
        'Closed-Form (OLS)': theta_closed,
        'Adam Optimizer': theta_adam,
        'Genetic Algorithm': theta_ga,
        'Simulated Annealing': theta_sa
    }
    
    hybrid_suite.plot_results(results_dict)
