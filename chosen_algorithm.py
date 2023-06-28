import numpy as np
import time
from tabulate import tabulate
from scipy.linalg import solve_triangular

# Hyper Parameters
MAX_ITERATIONS = 1000
ACCURACY = 1e-3 # ?
EPSILON = 1e-6
MU_START = 2
MU_FACTOR =5e-2


def check_matrix_is_PD(matrix):
    eigenvalues, _ = np.linalg.eigh(matrix)
    if all(eigenvalue > 0 for eigenvalue in eigenvalues):
        return True
    else:
        return False


def invert_lower_triangular(L):
    n = L.shape[0]
    L_inv = np.zeros_like(L)

    for i in range(n):
        L_inv[i, i] = 1 / L[i, i]
        for j in range(i-1, -1, -1):
            L_inv[i, j] = -np.dot(L[i, j:i], L_inv[j:i, j]) / L[i, i]

    return L_inv


def invert_PD_matrix(PD):
    if not check_matrix_is_PD(PD):
        print("3. Got non PD matrix:")
        print(PD)
        exit(1)

    L = np.real(np.linalg.cholesky(PD))
    L_inv = invert_lower_triangular(L)
    PD_inv = np.transpose(L_inv) @ L_inv

    return PD_inv


def calc_objective_func(X, A, mu):
    m = A.shape[0]

    val1 = -1 * np.log(np.linalg.det(X) + EPSILON)
    val2 = -1 * mu * np.sum([np.log(np.maximum(1 - np.dot(A[i], X @ A[i]), EPSILON)) for i in range(m)])

    return val1 + val2


def calc_grad_func(X, A, mu):
    m = A.shape[0]
    X_inv = invert_PD_matrix(X)

    val1 = -1 * np.transpose(X_inv)
    val2 = mu * np.sum([np.outer(A[i], A[i]) / np.maximum(1 - np.dot(A[i], X @ A[i]), EPSILON) for i in range(m)], axis=0)

    return val1 + val2


def calc_backtracking(X, func, grad_val_at_X):
    print("calc_backtracking 1")
    alpha = 0.1
    beta = 0.5
    step_size = 1/beta

    func_val = 1
    linear_approximation = 0
    num = 0
    while func_val > linear_approximation + EPSILON:
        step_size = beta * step_size
        matrix = project_to_PD(X - step_size * grad_val_at_X)

        if not check_matrix_is_PD:
            print("not pd matrix")
            exit(1)

        func_val = func(matrix)
        linear_approximation = func(X) - alpha * step_size * np.linalg.norm(grad_val_at_X) ** 2
        num += 1

        print(f"loop number: {num}\n"
              f"func_val = {func_val}\n"
              f"linear_approximation = {linear_approximation}\n"
              f"step_size = {step_size}\n")

    print("calc_backtracking 2")
    return step_size


def project_constraint(X, A):
    print("starting project_constraint")
    m = A.shape[0]

    k1 = 0
    for i in range(m):
        print("outer loop, k1={}".format(k1))
        k2 = 0
        while np.dot(A[i], np.linalg.inv(X) @ A[i]) > 1:
            #print("inner loop, k2={}".format(k2))
            X = X + 1e-3 * np.outer(A[i], A[i])
            k2 += 1

        k1 += 1

    print("ending project_constraint")
    return X


def project_to_PD(X):
    eigenvalues, eigenvectors = np.linalg.eigh(X)
    eigenvalues = np.maximum(eigenvalues, EPSILON)
    X_new = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    return X_new


def perform_grad_decent(X_start, A, mu):
    # Set variables
    X = X_start
    iteration = 0
    delta_x = 1.

    while iteration < MAX_ITERATIONS and delta_x > EPSILON:
        print(#f"iteration: {iteration + 1}\n"
            f"delta_x: {delta_x}")
        print("perform_grad_decent 1")
        # Compute the gradient of the objective function
        grad = calc_grad_func(X, A, mu)
        print("perform_grad_decent 2")
        # Compute step size using backtracking method
        learning_rate = calc_backtracking(X, lambda x: calc_objective_func(x, A, mu), grad)
        print("perform_grad_decent 3")
        # Update X using gradient descent
        X_new = X - learning_rate * grad
        print("perform_grad_decent 4")
        # Project X onto the set of positive definite matrices
        X_new = project_to_PD(X_new)
        print("perform_grad_decent 5")

        if not check_matrix_is_PD(X_new):
            print("1. Got non PD matrix:")
            print(X_new)
            exit(1)

        print("perform_grad_decent 6")
        # Project X onto the set of matrices satisfying the constraint
        X_new = project_constraint(X_new, A)
        print("perform_grad_decent 7")
        if not check_matrix_is_PD(X_new):
            print("2. Got non PD matrix:")
            print(X_new)
            exit(1)

        # Compute the distance between X_new and X
        print("perform_grad_decent 8")
        log_det_X = np.linalg.slogdet(X)[1]
        log_det_X_new = np.linalg.slogdet(X_new)[1]
        delta_x = np.abs(log_det_X - log_det_X_new)

        # Update the variables X and iteration
        print("perform_grad_decent 9")
        X = X_new
        iteration += 1

    print(f"Total number of iterations = {iteration}\n"
          f"delta_x: {delta_x}")
    return X


def solve(A: np.ndarray) -> np.ndarray:
    m, n = A.shape

    # Compute the covariance matrix of the data points
    covariance = np.cov(A.T)

    # Add a small positive constant to ensure positive definiteness
    X = covariance + EPSILON * np.eye(n)

    mu = MU_START
    ratio = m * mu

    while ratio > ACCURACY:
        print(f"mu: {mu}\n"
              f"ratio = {ratio}\n")

        # Perform gradient decent
        X = perform_grad_decent(X, A, mu)

        # Update the barrier parameter
        mu *= MU_FACTOR
        ratio = m * mu

    # Get the optimal solution and its properties
    det_X_inv = np.linalg.det(invert_PD_matrix(X) + EPSILON)
    logdet_X_inv = np.log(det_X_inv) if det_X_inv > 0 else -np.inf
    feasible = all([np.dot(A[i], invert_PD_matrix(X) @ A[i]) <= 1 for i in range(m)])

    return X, logdet_X_inv, feasible


def test_algo():
    A = np.load('Examples-20230617/blobs.1000.100.npy')
    A = np.load('Examples-20230617/checkerboard.50.4.npy')
    A = np.load('Examples-20230617/gaussian.2.5.npy')
    A = np.load('Examples-20230617/moons.50.2.npy')
    A = np.load('Examples-20230617/sparse.1000.10.npy')
    A = np.load('Examples-20230617/spiral.1000.20.npy')
    A = np.load('Examples-20230617/uniform.2.5.npy')
    A = np.load('Examples-20230617/uniform.10.10.npy')
    A = np.load('Examples-20230617/wave.50.4.npy')
    # A = np.array([[-0.79552, -0.04599, -0.17558, -0.61652, 0.26790],
    #               [0.79552, 0.04599, 0.17558, 0.61652, -0.26790]])

    results = []

    algorithm = solve

    print(f"Algorithm 1")
    start_time = time.time()
    X_opt, logdet_Xinv, feasible = algorithm(A)
    elapsed_time = time.time() - start_time
    results.append({
        "Algorithm": 1,
        "Time": elapsed_time,
        "logdet(X^-1)": logdet_Xinv,
        "Feasible": feasible,
        "is_pd": np.all(np.linalg.eigvals(X_opt) > 0),
        "scores": bool(
            np.all(np.einsum('...i,ij,...j->...', A, np.linalg.inv(X_opt), A) <= 1.))
    })

    # Print the results
    print(tabulate(results, headers="keys"))


if __name__ == "__main__":
    test_algo()
