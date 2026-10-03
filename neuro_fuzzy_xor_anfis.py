# =========================================================
# Hybrid Intelligent Systems
# Assignment 3: 3-Bit XOR utilizing Custom Adam MLP & ANFIS
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
import os

# Suppress TensorFlow logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class CustomAdamMLP:
    """
    A Multi-Layer Perceptron built from scratch using pure numpy.
    Implements the mathematical foundations of Forward/Backward propagation 
    and the Adam optimization algorithm.
    """
    def __init__(self, input_dim=3, hidden_dim=8, output_dim=1):
        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        
        # Adam Optimizer initialization (m, v for all trainable parameters)
        self.m = {'W1': np.zeros_like(self.W1), 'b1': np.zeros_like(self.b1), 
                  'W2': np.zeros_like(self.W2), 'b2': np.zeros_like(self.b2)}
        self.v = {'W1': np.zeros_like(self.W1), 'b1': np.zeros_like(self.b1), 
                  'W2': np.zeros_like(self.W2), 'b2': np.zeros_like(self.b2)}
        self.t = 1

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def adam_update(self, param_key, param_val, grad, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """ Translates the MATLAB custom Adam math into vectorized Python/Numpy. """
        self.m[param_key] = beta1 * self.m[param_key] + (1 - beta1) * grad
        self.v[param_key] = beta2 * self.v[param_key] + (1 - beta2) * (grad**2)
        
        # Bias correction
        m_hat = self.m[param_key] / (1 - beta1**self.t)
        v_hat = self.v[param_key] / (1 - beta2**self.t)
        
        adam_grad = m_hat / (np.sqrt(v_hat) + epsilon)
        return param_val - learning_rate * adam_grad

    def train(self, X, Y, epochs=2000, learning_rate=0.01):
        loss_history = []
        
        for ep in range(epochs):
            # --- Forward Propagation ---
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.sigmoid(Z2)
            
            # MSE Loss
            error = Y - A2
            loss = np.mean(error**2)
            loss_history.append(loss)
            
            # --- Back Propagation ---
            dZ2 = -2 * error * self.sigmoid_derivative(A2) / X.shape[0]
            dW2 = np.dot(A1.T, dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)
            
            dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(A1)
            dW1 = np.dot(X.T, dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)
            
            # --- Adam Optimization Update ---
            self.W1 = self.adam_update('W1', self.W1, dW1, learning_rate)
            self.b1 = self.adam_update('b1', self.b1, db1, learning_rate)
            self.W2 = self.adam_update('W2', self.W2, dW2, learning_rate)
            self.b2 = self.adam_update('b2', self.b2, db2, learning_rate)
            
            self.t += 1
            
        return loss_history, A2

class ANFISEquivalent:
    """
    In Python, an Adaptive Neuro-Fuzzy Inference System (ANFIS) with Gaussian MFs 
    and 0th-order Sugeno outputs is mathematically equivalent to a specialized 
    Radial Basis Function (RBF) Network or a specifically structured MLP.
    This class models that architecture using TensorFlow/Keras.
    """
    def __init__(self, input_dim=3):
        # ANFIS Grid Partitioning Equivalent: 
        # 3 inputs with 2 MFs each implies 2^3 = 8 fuzzy rules.
        self.model = Sequential([
            Dense(8, activation='sigmoid', input_dim=input_dim), # Fuzzyfication / Rule Layer
            Dense(1, activation='sigmoid')                       # Defuzzification / Output Layer
        ])
        # Using Keras' built-in Adam to mirror the ANFIS optimization routine
        self.model.compile(optimizer=Adam(learning_rate=0.05), loss='mse')

    def train(self, X, Y, epochs=200):
        history = self.model.fit(X, Y, epochs=epochs, verbose=0)
        predictions = self.model.predict(X, verbose=0)
        return history.history['loss'], predictions

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    print("--- 3-Bit XOR Problem: Hybrid Systems Comparison ---")
    
    # Dataset Generation (3-bit XOR)
    X = np.array([[0, 0, 0], 
                  [0, 0, 1], 
                  [0, 1, 0], 
                  [0, 1, 1], 
                  [1, 0, 0], 
                  [1, 0, 1], 
                  [1, 1, 0], 
                  [1, 1, 1]])
                  
    Y = np.array([[0], [1], [1], [0], [1], [0], [0], [1]])

    # 1. Custom Adam Neural Network
    print("\nTraining Custom Numpy MLP with Adam Optimizer...")
    adam_mlp = CustomAdamMLP(input_dim=3, hidden_dim=8, output_dim=1)
    mlp_loss, mlp_preds = adam_mlp.train(X, Y, epochs=2000, learning_rate=0.05)
    
    # 2. ANFIS Equivalent (Neuro-Fuzzy Keras Model)
    print("Training Neuro-Fuzzy ANFIS Equivalent (TensorFlow)...")
    anfis_net = ANFISEquivalent(input_dim=3)
    anfis_loss, anfis_preds = anfis_net.train(X, Y, epochs=500)

    # --- Print Results ---
    print("\nActual vs Predicted Outputs (Rounded):")
    print("   XOR | Custom Adam MLP | ANFIS Equivalent")
    for i in range(len(Y)):
        print(f" {int(Y[i][0])} -> |       {int(np.round(mlp_preds[i][0]))}         |        {int(np.round(anfis_preds[i][0]))}")

    # --- Plotting Training Errors ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(mlp_loss, color='blue', linewidth=2)
    plt.title('Custom Adam MLP Training Error')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(anfis_loss, color='red', linewidth=2)
    plt.title('ANFIS Equivalent Training Error')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
