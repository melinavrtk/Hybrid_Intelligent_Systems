# =========================================================
# Hybrid Intelligent Systems
# Assignment 1: Fuzzy Logic Controller for Lighting
# =========================================================

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

class FuzzyLightingController:
    """
    A Fuzzy Logic Controller that adjusts a bulb's brightness based on 
    ambient brightness and its rate of change. Implements fuzzy set 
    intersections and Mamdani inference rules from scratch.
    """

    def __init__(self):
        # 1. Define Universes of Discourse (Χώροι Αναφοράς)
        self.x_brightness = np.arange(0, 100.1, 0.1)
        self.x_roc = np.arange(-10, 10.1, 0.1)       # Rate of change
        self.x_bulb = np.arange(0, 10.1, 0.1)        # Bulb brightness

        # 2. Define Membership Functions (Συναρτήσεις Συμμετοχής)
        # A) Ambient Brightness
        self.b_low = fuzz.trapmf(self.x_brightness, [0, 0, 10, 20])
        self.b_med = fuzz.trimf(self.x_brightness, [15, 40, 65])
        self.b_high = fuzz.trapmf(self.x_brightness, [60, 80, 100, 100])

        # B) Rate of Change
        self.roc_low = fuzz.trapmf(self.x_roc, [-10, -10, -5, -1])
        self.roc_none = fuzz.trimf(self.x_roc, [-2, 0, 2])
        self.roc_high = fuzz.trapmf(self.x_roc, [1, 5, 10, 10])

        # C) Bulb Brightness (Output)
        self.bulb_low = fuzz.trapmf(self.x_bulb, [0, 0, 2, 5])
        self.bulb_med = fuzz.trimf(self.x_bulb, [3, 5, 7])
        self.bulb_high = fuzz.trapmf(self.x_bulb, [5, 8, 10, 10])

    def evaluate_system(self):
        """ Evaluates the fuzzy rules based on fuzzy inputs. """
        
        # --- Define Fuzzy Inputs (Είσοδοι Συστήματος) ---
        # Input 1: Ambient Brightness (Trapezoidal fuzzy set)
        b_in = fuzz.trapmf(self.x_brightness, [10, 30, 40, 60])
        
        # Input 2: Rate of Change (Sigmoid fuzzy set)
        # Note: scikit-fuzzy sigmf takes (universe, center, width multiplier)
        roc_in = fuzz.sigmf(self.x_roc, 5, 0.5)

        # --- Calculate Activation Weights (Βάρη Ενεργοποίησης) ---
        # using np.fmin for element-wise minimum (intersection) and np.max for the peak
        w_b_low = np.max(np.fmin(b_in, self.b_low))
        w_b_med = np.max(np.fmin(b_in, self.b_med))
        w_b_high = np.max(np.fmin(b_in, self.b_high))

        w_roc_low = np.max(np.fmin(roc_in, self.roc_low))
        w_roc_none = np.max(np.fmin(roc_in, self.roc_none))
        w_roc_high = np.max(np.fmin(roc_in, self.roc_high))

        # --- Evaluate Fuzzy Rules (Κανόνες Mamdani) ---
        
        # Rule 1: IF Brightness is Low AND Rate of Change is Low (Negative) 
        #         THEN Bulb Brightness is High (Squared modifier for "Very High")
        activation_rule_1 = np.min([w_b_low, w_roc_low**2])
        out_bulb_rule_1 = np.fmin(activation_rule_1, self.bulb_high**2)

        # Rule 2: IF Brightness is Med THEN Bulb Brightness is Med
        activation_rule_2 = w_b_med
        out_bulb_rule_2 = np.fmin(activation_rule_2, self.bulb_med)
        
        # Rule 3: IF Brightness is High THEN Bulb Brightness is Low
        activation_rule_3 = w_b_high
        out_bulb_rule_3 = np.fmin(activation_rule_3, self.bulb_low)

        # --- Aggregation and Defuzzification (Αποσαφήνιση) ---
        # Combine all rule outputs
        aggregated_output = np.fmax(out_bulb_rule_1, np.fmax(out_bulb_rule_2, out_bulb_rule_3))
        
        # Defuzzify using Centroid method
        bulb_crisp_value = fuzz.defuzz(self.x_bulb, aggregated_output, 'centroid')
        
        print(f"--- Fuzzy Controller Results ---")
        print(f"Rule 1 Activation: {activation_rule_1:.3f}")
        print(f"Rule 2 Activation: {activation_rule_2:.3f}")
        print(f"Rule 3 Activation: {activation_rule_3:.3f}")
        print(f"\n=> Defuzzified Bulb Brightness: {bulb_crisp_value:.2f} / 10.0")

        self.visualize(aggregated_output, bulb_crisp_value)

    def visualize(self, aggregated, crisp_val):
        """ Visualizes the output membership functions and the defuzzified result. """
        plt.figure(figsize=(8, 5))
        
        # Plot output membership functions
        plt.plot(self.x_bulb, self.bulb_low, 'b', linewidth=1.5, label='Low')
        plt.plot(self.x_bulb, self.bulb_med, 'g', linewidth=1.5, label='Medium')
        plt.plot(self.x_bulb, self.bulb_high, 'r', linewidth=1.5, label='High')
        
        # Fill the aggregated area
        plt.fill_between(self.x_bulb, 0, aggregated, facecolor='orange', alpha=0.5, label='Aggregated Output')
        
        # Plot the crisp result line
        plt.vlines(crisp_val, 0, 1, color='k', linestyle='--', linewidth=2, label=f'Crisp Value ({crisp_val:.2f})')
        
        plt.title('Fuzzy Lighting Control Output')
        plt.xlabel('Bulb Brightness (0-10)')
        plt.ylabel('Membership Degree')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    controller = FuzzyLightingController()
    controller.evaluate_system()
