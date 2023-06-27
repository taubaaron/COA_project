import numpy as np
import time
from tabulate import tabulate

# Hyper Parameters
MAX_ITERATIONS = 1000
EPSILON = 1e-6
MU_FACTOR = 0.9
MU_MIN = 1e-8
STEP_SIZE = 0.5


def solve_optimization_5(A):
    n, d = A.shape

    # Compute the covariance matrix of the data points
    covariance = np.cov(A.T)

    # Add a small positive constant to ensure positive definiteness
    epsilon = EPSILON
    X = covariance + epsilon * np.eye(d)

    # Set up the optimization loop
    max_iterations = MAX_ITERATIONS
    learning_rate = STEP_SIZE
    mu = 1.0  # Initial barrier parameter

    for iteration in range(max_iterations):
        print(f"iteration: {iteration}")
        X_inv = np.linalg.inv(X)

        # Compute the objective function and constraints
        objective = np.log(np.linalg.det(X_inv) + epsilon) - mu * np.sum(
            [np.log(np.maximum(1 - np.dot(A[i], X_inv @ A[i]), epsilon)) for i in range(n)])

        # Compute the gradient of the objective function
        grad = np.linalg.inv(X_inv).T - mu * np.sum(
            [np.outer(A[i], A[i]) / np.maximum(1 - np.dot(A[i], X_inv @ A[i]), epsilon) for i in range(n)], axis=0)

        # Update X using gradient descent
        X_new = X - learning_rate * grad

        # Project X onto the set of positive definite matrices
        eigenvalues, eigenvectors = np.linalg.eig(X_new)
        eigenvalues = np.maximum(eigenvalues, epsilon)
        X_new = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Project X onto the set of matrices satisfying the constraint
        X_new = project_constraint(X_new, A)

        # Check convergence
        if np.allclose(X, X_new):
            print("break: convergence")
            break

        X = X_new

        # Check if the barrier parameter is small enough
        if mu < MU_MIN:
            print("break: mu < MU_MIN")
            break

        # Update the barrier parameter
        mu *= MU_FACTOR

    # Get the optimal solution and its properties
    det_X_inv = np.linalg.det(np.linalg.inv(X) + epsilon)
    logdet_Xinv = np.log(det_X_inv) if det_X_inv > 0 else -np.inf
    feasible = all([np.dot(A[i], np.linalg.inv(X) @ A[i]) <= 1 for i in range(n)])

    return X, logdet_Xinv, feasible


def project_constraint(X, A):
    n = A.shape[0]

    for i in range(n):
        while np.dot(A[i], np.linalg.inv(X) @ A[i]) > 1:
            X = X + 1e-3 * np.outer(A[i], A[i])

    return X


def test_algo():
    matrix_files = [
        'Examples-20230617/blobs.100.10.npy',
        'Examples-20230617/checkerboard.50.4.npy',
        'Examples-20230617/gaussian.2.5.npy',
        'Examples-20230617/moons.50.2.npy',
        'Examples-20230617/sparse.1000.10.npy',
        'Examples-20230617/spiral.1000.20.npy',
        'Examples-20230617/uniform.2.5.npy',
        'Examples-20230617/uniform.10.10.npy',
        'Examples-20230617/wave.50.4.npy'
    ]

    results = []
    algorithm = solve_optimization_5

    for i, matrix_file in enumerate(matrix_files):
        A = np.load(matrix_file)

        print(f"\n\nRunning Algorithm on Matrix {i+1} - {matrix_file}")
        start_time = time.time()
        X_opt, logdet_Xinv, feasible = algorithm(A)
        elapsed_time = time.time() - start_time

        result = {
            "Matrix": matrix_file,
            "Time": elapsed_time,
            "logdet(X^-1)": logdet_Xinv,
            "Feasible": feasible,
            "is_pd": np.all(np.linalg.eigvals(X_opt) > 0),
            "scores": bool(np.all(np.einsum('...i,ij,...j->...', A, np.linalg.inv(X_opt), A) <= 1.))
        }

        results.append(result)

    # Print the results
    print(tabulate(results, headers="keys"))



if __name__ == "__main__":
    test_algo()
