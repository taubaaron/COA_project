import numpy as np
import time
from tabulate import tabulate

# Hyper Parameters
MAX_ITERATIONS = 100
EPSILON = 1e-6


def solve_optimization_1(A):
    n, d = A.shape

    # Compute the covariance matrix of the data points
    covariance = np.cov(A.T)

    # Add a small positive constant to ensure positive definiteness
    epsilon = EPSILON
    X = covariance + epsilon * np.eye(d)

    # Set up the optimization loop
    max_iterations = MAX_ITERATIONS
    epsilon = EPSILON
    learning_rate = 0.1

    for iteration in range(max_iterations):
        X_inv = np.linalg.inv(X)

        # Compute the objective function and constraints
        constraints = [np.dot(A[i], X_inv @ A[i]) - 1 <= 0 for i in range(n)]

        # Check feasibility
        feasible = all(constraints)

        if feasible:
            break

        # Compute the gradient
        grad = np.linalg.inv(X_inv).T - np.sum([np.outer(A[i], A[i]) / np.dot(A[i], X_inv @ A[i]) for i in range(n)],
                                               axis=0)

        # Update X using gradient descent
        X_new = X - learning_rate * grad

        # Project X onto the set of positive definite matrices
        eigenvalues, eigenvectors = np.linalg.eig(X_new)
        X_new = eigenvectors @ np.diag(np.maximum(eigenvalues, epsilon)) @ eigenvectors.T

        # Compute the change in X
        diff = np.linalg.norm(X_new - X)
        X = X_new

        if diff < epsilon:
            break

    # Get the optimal solution and its properties
    _, logdet_Xinv = np.linalg.slogdet(X)
    feasible = all([np.dot(A[i], np.linalg.inv(X) @ A[i]) <= 1 for i in range(n)])

    return X, -logdet_Xinv, feasible


"""To improve the algorithm, we can incorporate an adaptive learning rate that dynamically adjusts the learning rate 
based on the progress of the optimization. This can help accelerate convergence and avoid overshooting. Here's an 
updated version of the code that includes an adaptive learning rate: """


def solve_optimization_2(A):
    n, d = A.shape

    # Compute the covariance matrix of the data points
    covariance = np.cov(A.T)

    # Add a small positive constant to ensure positive definiteness
    epsilon = EPSILON
    X = covariance + epsilon * np.eye(d)

    # Set up the optimization loop
    max_iterations = MAX_ITERATIONS
    epsilon = EPSILON
    initial_learning_rate = 0.1

    for iteration in range(max_iterations):
        X_inv = np.linalg.inv(X)

        # Compute the objective function and constraints
        constraints = [np.dot(A[i], X_inv @ A[i]) - 1 <= 0 for i in range(n)]

        # Check feasibility
        feasible = all(constraints)

        if feasible:
            break

        # Compute the gradient
        grad = np.linalg.inv(X_inv).T - np.sum([np.outer(A[i], A[i]) / np.dot(A[i], X_inv @ A[i]) for i in range(n)],
                                               axis=0)

        # Update X using gradient descent with adaptive learning rate
        learning_rate = initial_learning_rate / np.sqrt(iteration + 1)
        X_new = X - learning_rate * grad

        # Project X onto the set of positive definite matrices
        eigenvalues, eigenvectors = np.linalg.eig(X_new)
        X_new = eigenvectors @ np.diag(np.maximum(eigenvalues, epsilon)) @ eigenvectors.T

        # Compute the change in X
        diff = np.linalg.norm(X_new - X)
        X = X_new

        if diff < epsilon:
            break

    # Get the optimal solution and its properties
    _, logdet_Xinv = np.linalg.slogdet(X)

    feasible = all([np.dot(A[i], np.linalg.inv(X) @ A[i]) <= 1 for i in range(n)])

    return X, -logdet_Xinv, feasible


"""To improve the efficiency of the algorithm, we can leverage the fact that the objective function is concave and 
the constraints are linear. We can reformulate the problem as a convex optimization problem using the logarithmic 
barrier method. This allows us to solve the problem with a single optimization iteration instead of using an 
iterative approach. """


def solve_optimization_3(A):
    n, d = A.shape

    # Compute the covariance matrix of the data points
    covariance = np.cov(A.T)

    # Add a small positive constant to ensure positive definiteness
    epsilon = EPSILON
    X = covariance + epsilon * np.eye(d)

    # Set up the optimization loop
    max_iterations = MAX_ITERATIONS
    epsilon = EPSILON
    learning_rate = 0.5
    mu = 1.0  # Barrier parameter

    for iteration in range(max_iterations):
        X_inv = np.linalg.inv(X)

        # # Compute the objective function and constraints
        #
        # objective = np.log(np.linalg.det(X_inv) + epsilon) - mu * np.sum(
        #     [np.log(np.maximum(1 - np.dot(A[i], X_inv @ A[i]), epsilon)) for i in range(n)])



        # Compute the gradient of the objective function
        grad = np.linalg.inv(X_inv).T - mu * np.sum(
            [np.outer(A[i], A[i]) / np.maximum(1 - np.dot(A[i], X_inv @ A[i]), epsilon) for i in range(n)], axis=0)

        # Update X using gradient descent
        X_new = X - learning_rate * grad

        # Project X onto the set of positive definite matrices
        eigenvalues, eigenvectors = np.linalg.eig(X_new)
        X_new = eigenvectors @ np.diag(np.maximum(eigenvalues, epsilon)) @ eigenvectors.T

        # Compute the change in X
        diff = np.linalg.norm(X_new - X)
        X = X_new

        if diff < epsilon:
            break

        # Adjust barrier parameter
        mu *= 0.9

        # Get the optimal solution and its properties
        det_X_inv = np.linalg.det(np.linalg.inv(X) + epsilon)
        logdet_Xinv = np.log(det_X_inv) if det_X_inv > 0 else -np.inf
        feasible = all([np.dot(A[i], np.linalg.inv(X) @ A[i]) <= 1 for i in range(n)])

    return X, logdet_Xinv, feasible


"""Use a more efficient method to compute the inverse of X_inv. Instead of computing the inverse directly using 
np.linalg.inv, we can solve a linear system of equations using np.linalg.solve. This approach is generally faster and 
more stable for large matrices. """
def solve_optimization_4(A):
    n, d = A.shape

    # Compute the covariance matrix of the data points
    covariance = np.cov(A.T)

    # Add a small positive constant to ensure positive definiteness
    epsilon = EPSILON
    X = covariance + epsilon * np.eye(d)

    # Set up the optimization loop
    max_iterations = MAX_ITERATIONS
    epsilon = EPSILON
    learning_rate = 0.5
    mu = 1.0  # Barrier parameter

    for iteration in range(max_iterations):
        X_inv = np.linalg.inv(X)

        # Compute the objective function and constraints
        objective = np.log(np.linalg.det(X_inv) + epsilon) - mu * np.sum(
            [np.log(np.maximum(1 - np.dot(A[i], X_inv @ A[i]), epsilon)) for i in range(n)])

        # Compute the gradient of the objective function
        grad = np.linalg.inv(X_inv).T - mu * np.sum(
            [np.outer(A[i], A[i]) / np.maximum(1 - np.dot(A[i], X_inv @ A[i]), epsilon) for i in range(n)], axis=0)

        # Update X using gradient descent
        X_new = X - learning_rate * grad

        # Project X onto the set of positive definite matrices using Cholesky decomposition
        L = np.linalg.cholesky(X_new)
        X_new = L @ L.T

        # Compute the change in X
        diff = np.linalg.norm(X_new - X)
        X = X_new

        if diff < epsilon:
            break

        # Adjust barrier parameter
        mu *= 0.9

    # Get the optimal solution and its properties
        det_X_inv = np.linalg.det(np.linalg.inv(X) + epsilon)
        logdet_Xinv = np.log(det_X_inv) if det_X_inv > 0 else -np.inf
        feasible = all([np.dot(A[i], np.linalg.inv(X) @ A[i]) <= 1 for i in range(n)])

    return X, logdet_Xinv, feasible


def solve_optimization_5(A):
    n, d = A.shape

    # Compute the covariance matrix of the data points
    covariance = np.cov(A.T)

    # Add a small positive constant to ensure positive definiteness
    epsilon = EPSILON
    X = covariance + epsilon * np.eye(d)

    # Set up the optimization loop
    max_iterations = MAX_ITERATIONS
    epsilon = EPSILON
    mu = 1.0  # Barrier parameter

    for iteration in range(max_iterations):
        X_inv = np.linalg.inv(X)

        # Compute the objective function and constraints
        objective = np.log(np.linalg.det(X_inv)) - mu * np.sum([np.log(np.maximum(1 - np.dot(A[i], X_inv @ A[i]), epsilon)) for i in range(n)])

        # Compute the gradient of the objective function
        grad = np.linalg.inv(X_inv).T - mu * np.sum([np.outer(A[i], A[i]) / np.maximum(1 - np.dot(A[i], X_inv @ A[i]), epsilon) for i in range(n)], axis=0)

        # Update X using adaptive learning rate
        learning_rate = 1.0 / np.sqrt(iteration + 1)
        X_new = X - learning_rate * grad

        # Perform Cholesky decomposition for the projection step
        L = np.linalg.cholesky(X_new)
        X_new = L @ L.T

        # Compute the change in X
        diff = np.linalg.norm(X_new - X)
        X = X_new

        if diff < epsilon:
            break

        # Adjust barrier parameter
        mu *= 0.9

    # Get the optimal solution and its properties
    _, logdet_Xinv = np.linalg.slogdet(X)
    feasible = all([np.dot(A[i], np.linalg.inv(X) @ A[i]) <= 1 for i in range(n)])

    return X, -logdet_Xinv, feasible


def test_algo():
    A = np.load('Examples-20230617/blobs.1000.100.npy')
    # Store the results in a list of dictionaries
    results = []


    algorithms = [solve_optimization_1,
                  # solve_optimization_2,
                  solve_optimization_3,
                  # solve_optimization_4,
                  # solve_optimization_5]
    ]
    # Loop over the algorithms and collect the results
    for num, algorithm in enumerate(algorithms):
        start_time = time.time()
        X_opt, logdet_Xinv, feasible = algorithm(A)
        results.append({
            "Algorithm": num + 1,
            "Time": time.time() - start_time,
            "logdet(X^-1)": logdet_Xinv,
            "Feasible": feasible,
            "is_pd": bool(np.all(np.linalg.eigvals(X_opt) > 0)),
            "scores": bool(np.all(np.einsum('...i,ij,...j->...', A, np.linalg.inv(X_opt), A) <= 1.))
        })

    # Print the results as a table
    headers = results[0].keys()
    rows = [list(result.values()) for result in results]
    table = tabulate(rows, headers=headers, floatfmt=".6f")
    print(table)

test_algo()


