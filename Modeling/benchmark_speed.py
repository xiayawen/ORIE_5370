"""
Micro-benchmark: Analytical Closed-Form MVO vs. CVXPYLayers
Compares the computational speed of our pure PyTorch equality-constrained 
MVO layer against a standard numerical interior-point solver.
"""

import time
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# Set random seed for reproducibility
torch.manual_seed(42)

# --- Hyperparameters ---
BATCH_SIZE = 64
N_ASSETS = 100
DELTA = 50.0
ITERATIONS = 50

print(f"Setting up benchmark for Batch Size: {BATCH_SIZE}, Assets: {N_ASSETS}...")

# --- 1. Generate Synthetic Data ---
# Random predictions (y_hat)
y_hat = torch.randn(BATCH_SIZE, N_ASSETS, requires_grad=True)

# Random symmetric positive-definite covariance matrices (V)
A = torch.randn(BATCH_SIZE, N_ASSETS, N_ASSETS)
V = torch.bmm(A, A.transpose(1, 2)) + torch.eye(N_ASSETS).unsqueeze(0) * 0.1
# Ensure perfect symmetry
V = (V + V.transpose(1, 2)) / 2.0

# ** THE DPP TRICK **
# For CVXPYLayers to be differentiable (DPP-compliant), we cannot pass V directly 
# into a quadratic form. Instead, we calculate the Cholesky decomposition (V = L L^T)
# and use it to calculate the squared norm ||L^T z||_2^2.
L = torch.linalg.cholesky(V)


# --- 2. Define Method A: Our Analytical Closed-Form Layer ---
def analytical_mvo(y_hat_batch, V_batch, delta):
    B, N = y_hat_batch.shape
    V_inv = torch.linalg.inv(V_batch)
    ones = torch.ones(N, 1, dtype=y_hat_batch.dtype, device=y_hat_batch.device)
    
    y_col = y_hat_batch.unsqueeze(2) # (B, N, 1)
    
    # Calculate lambda
    num = torch.bmm(ones.T.unsqueeze(0).expand(B, -1, -1), torch.bmm(V_inv, y_col))
    den = torch.bmm(ones.T.unsqueeze(0).expand(B, -1, -1), torch.matmul(V_inv, ones))
    lam = (num - delta) / den
    
    # Calculate optimal z*
    z = (1.0 / delta) * torch.bmm(V_inv, y_col - lam * ones.unsqueeze(0))
    return z.squeeze(2)


# --- 3. Define Method B: CVXPYLayers Numerical Solver ---
z_var = cp.Variable(N_ASSETS)
y_param = cp.Parameter(N_ASSETS)
L_param = cp.Parameter((N_ASSETS, N_ASSETS)) # Pass L instead of V

# Objective: Minimize -y^T z + (delta/2) * ||L^T z||_2^2
objective = cp.Minimize(-y_param.T @ z_var + (DELTA / 2.0) * cp.sum_squares(L_param.T @ z_var))
constraints = [cp.sum(z_var) == 1.0]
prob = cp.Problem(objective, constraints)

# Create the PyTorch layer
cvxpy_layer = CvxpyLayer(prob, parameters=[y_param, L_param], variables=[z_var])


# --- 4. Benchmark Execution ---

# Warmup
_ = analytical_mvo(y_hat, V, DELTA)
try:
    _ = cvxpy_layer(y_hat, L, solver_args={'max_iters': 10000})
except Exception:
    pass 

print("\n--- Running Benchmark ---")

# --- Time Analytical Method ---
start_time = time.perf_counter()
for _ in range(ITERATIONS):
    # Forward pass
    z_analytical = analytical_mvo(y_hat, V, DELTA)
    # Dummy loss to force backward pass (gradient calculation)
    loss = z_analytical.sum()
    loss.backward(retain_graph=True)
    
    # Zero gradients
    y_hat.grad = None
analytical_time = time.perf_counter() - start_time
print(f"Analytical Closed-Form Time : {analytical_time:.4f} seconds")

# --- Time CVXPYLayers Method ---
start_time = time.perf_counter()
for _ in range(ITERATIONS):
    # Forward pass (Using L, not V)
    z_cvxpy, = cvxpy_layer(y_hat, L)
    # Dummy loss to force backward pass
    loss = z_cvxpy.sum()
    loss.backward(retain_graph=True)
    
    # Zero gradients
    y_hat.grad = None
cvxpy_time = time.perf_counter() - start_time
print(f"CVXPYLayers Numerical Time  : {cvxpy_time:.4f} seconds")

# --- Results ---
speedup = cvxpy_time / analytical_time
print("\n--- Conclusion ---")
print(f"Our Analytical Method is {speedup:.1f}x faster than CVXPYLayers.")
print("This explains why we can train Neural Networks end-to-end without massive computational bottlenecks.")