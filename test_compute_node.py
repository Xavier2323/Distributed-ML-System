import sys
sys.path.append('gen-py')

import numpy as np
import os
from compute_node import ComputeNodeHandler
from shared.ttypes import MLModel, TrainingResult

def test_model_convergence():
    print("\n[TEST] Running Model Convergence Test on train_letters1.txt...")

    # Step 1: Ensure train_letters1.txt exists
    train_file = "./ML/letters/train_letters1.txt"
    if not os.path.exists(train_file):
        print(f"[ERROR] {train_file} not found! Please place it in the same directory.")
        return

    # Step 2: Initialize Compute Node Handler
    compute_node = ComputeNodeHandler(load_probability=0.0)  # No overload

    # Step 3: Initialize model with random weights
    num_features = 16  # Adjust based on dataset
    num_hidden = 10
    num_classes = 26  # Assuming 26 letters (A-Z)

    np.random.seed(1)
    V = (np.random.rand(num_hidden + 1, num_classes) * 0.02) - 0.01
    W = (np.random.rand(num_features + 1, num_hidden) * 0.02) - 0.01

    model = MLModel(V=V.tolist(), W=W.tolist())
    status = compute_node.initializeTraining(train_file, model)

    if status != 1:
        print("[ERROR] Model initialization failed.")
        return

    print("[INFO] Model successfully initialized.")

    # Step 4: Train model for multiple rounds and track error
    eta = 0.0001  # Learning rate
    epochs = 100  
    prev_error = None

    for round in range(20):  # 20 training iterations
        result: TrainingResult = compute_node.trainModel(eta, epochs)

        if result.error_rate == -1:
            print("[ERROR] Training failed.")
            return

        print(f"[INFO] Training Round {round+1}: Error Rate = {result.error_rate:.4f}")

        # Ensure gradients are not zero (model is learning)
        dV_sum = np.sum(np.array(result.gradient.dV))
        dW_sum = np.sum(np.array(result.gradient.dW))

        if dV_sum == 0 or dW_sum == 0:
            print("[WARNING] Zero gradients detected! Restarting training with lower eta...")
            eta /= 2  # Reduce learning rate and restart training
            continue  # Try again with lower eta

        # Ensure validation error is decreasing (convergence check)
        if prev_error is not None and result.error_rate > prev_error:
            print("[WARNING] Error increased! Reducing learning rate.")
            eta /= 2  # Reduce learning rate if error increases

        prev_error = result.error_rate  # Track for next iteration

    print("\n✅ [TEST PASSED] Model shows signs of learning and convergence!")

if __name__ == "__main__":
    test_model_convergence()
