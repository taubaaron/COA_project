#
# import numpy as np
#
# def gradient_descent_algo(A):
#     def solve(A: np.ndarray) -> np.ndarray:
#         def gradient_descent(A, learning_rate=0.1, tol=1e-6, max_iter=1000):
#             m, n = A.shape  # Dimensions of the matrix
#
#             # Initial guess for the primal variables
#             X_inv = np.eye(n)
#
#             for iteration in range(max_iter):
#                 X = np.linalg.inv(X_inv)
#                 grad_phi = X - (A.T / np.diagonal(A @ X @ A.T)) @ A
#
#                 # Project the primal variable onto the set of positive definite matrices
#                 X_inv = project_pd(X_inv - learning_rate * grad_phi)
#
#                 sign, logdet = np.linalg.slogdet(X)
#
#                 print(f"grad_phi: {np.linalg.norm(grad_phi)}\n"
#                       f"Iter: {iteration}\n"
#                       f"Sign: {sign}\n"
#                       f"Log-det: {logdet}")
#                 # Check convergence condition
#                 if np.linalg.norm(grad_phi, ord='fro') <= tol:
#                     break
#
#             return np.linalg.inv(X_inv)
#
#         # Project the matrix onto the set of positive definite matrices
#         def project_pd(X):
#             eig_vals, eig_vecs = np.linalg.eigh(X)
#             eig_vals = np.maximum(eig_vals, 1e-6)  # Ensure positive eigenvalues
#             return eig_vecs @ np.diag(eig_vals) @ eig_vecs.T
#
#         # Call the gradient_descent function
#         X_opt = gradient_descent(A)
#         return X_opt
#
#     x_opt = solve(A)
#     return x_opt
#
#
# def projected_gradient_descent_algo():
#
#     def solve(A: np.ndarray) -> np.ndarray:
#         def projected_gradient_descent(A, learning_rate=0.1, tol=1e-6, max_iter=1000):
#             m, n = A.shape  # Dimensions of the matrix
#
#             # Initialize primal variable
#             X_inv = np.eye(n)
#
#             for iteration in range(max_iter):
#                 X = np.linalg.inv(X_inv)
#                 grad_phi = X - (A.T / np.diagonal(A @ X @ A.T)) @ A
#
#                 # Update primal variable with projected gradient descent step
#                 X_inv -= learning_rate * grad_phi
#                 X_inv = project_pd(X_inv)
#
#                 sign, logdet = np.linalg.slogdet(X)
#
#                 print("grad_phi: ", np.linalg.norm(grad_phi))
#                 print("Iter: ", iteration)
#                 print("Sign: ", sign)
#                 print("Log-det: ", logdet)
#
#                 # Check convergence condition
#                 if np.linalg.norm(grad_phi, ord='fro') <= tol:
#                     break
#
#             return np.linalg.inv(X_inv)
#
#         def project_pd(X):
#             eig_vals, eig_vecs = np.linalg.eigh(X)
#             eig_vals = np.maximum(eig_vals, 1e-6)  # Ensure positive eigenvalues
#             return eig_vecs @ np.diag(eig_vals) @ eig_vecs.T
#
#         X_opt = projected_gradient_descent(A)
#
#         return X_opt
#
#     # Example usage
#     A = np.load('Examples-20230617/blobs.100.10.npy')
#
#     X_opt = solve(A)
#
# import numpy as np
#
# def solve_convexity_optimization(m, a, tol=1e-6, max_iter=100):
#     n = len(a[0])
#
#     # Initial solution
#     X = np.eye(n)
#
#     # Newton's method
#     for iter_count in range(max_iter):
#         X_inv = np.linalg.inv(X)
#         S = np.linalg.inv(X_inv)  # Schur complement
#
#         # Compute the gradient and Hessian
#         gradient = np.zeros((n, n))
#         hessian = np.zeros((n, n))
#
#         for i in range(m):
#             A = np.outer(a[i], a[i])
#             term1 = np.trace(S @ A @ S @ X_inv)
#             term2 = np.trace(S @ A)
#             term3 = np.trace(S @ A @ X_inv)
#             term4 = np.trace(A @ S @ X_inv)
#
#             gradient += term1 * X_inv + term2 * X - term3 * X_inv - term4 * X_inv
#
#             hessian += term1 * np.kron(X_inv, X_inv) + term2 * np.kron(X, X)
#             hessian -= term3 * np.kron(X_inv, np.eye(n))
#             hessian -= term4 * np.kron(np.eye(n), X_inv)
#
#         gradient /= m
#         hessian /= m
#
#         # Compute the Newton step
#         step = np.linalg.solve(hessian, -gradient)
#
#         # Line search
#         t = 1.0
#         while t > 1e-10:
#             X_new = X + t * step
#             if np.all(np.linalg.eigvals(X_new) > 0):
#                 break
#             t *= 0.5
#
#         # Check convergence
#         if np.linalg.norm(X_new - X, 'fro') / n < tol:
#             break
#
#         X = X_new
#
#     return X
#
# # Example usage
# m = 2
# a = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
#
# X_optimal = solve_convexity_optimization(m, a)
# print("Optimal X:")
# print(X_optimal)
#
#
# if __name__ == "__main__":
#     A=np.load('Examples-20230617/blobs.100.10.npy')
#     X_star = gradient_descent_algo(A)
#     print(X_star)
#     print(X_star.shape)
#     eigenvalues = np.linalg.eigvals(X_star)
#     is_pd = np.all(eigenvalues > 0)
#     scores = np.einsum('...i,ij,...j->...', A, np.linalg.inv(X_star), A)
#
#     print(is_pd, np.all(scores <= 1.))
#
#     # projected_gradient_descent_algo()
#

import numpy as np

def solve_convexity_optimization(m, n, a):
    # Construct the constraint matrix A
    A = np.zeros((m, n, n))
    for i in range(m):
        A[i] = np.outer(a[i], a[i])

    # Initialize X with a positive definite matrix
    X = np.eye(n)

    # Set up the optimization loop
    max_iterations = 100
    epsilon = 1e-6

    for iteration in range(max_iterations):
        X_inv = np.linalg.inv(X)

        # Compute the gradient
        grad = np.einsum('ijk,kl,ijl->il', A, X_inv, A)
        grad /= np.trace(grad)  # Normalization

        # Line search
        alpha = 1.0
        while True:
            X_new = X - alpha * np.linalg.inv(X) * np.max(np.einsum('...i,ij,...j->...', A, X, A))

            # Check constraints
            valid = np.all(np.einsum('ijk,kl->ij', A, X_new) <= 1.0)

            if valid:
                break

            alpha *= 0.5

        diff = np.linalg.norm(X_new - X)
        X = X_new

        if diff < epsilon:
            break

    return X

# Example usage
m = 2  # Number of constraints
n = 3  # Dimension of X
a = [np.array([1, 2, 3]), np.array([4, 5, 6])]  # Constraint vectors

X_optimal = solve_convexity_optimization(m, n, a)
print(f"Optimal X: \n{X_optimal}")


# A=np.load('Examples-20230617/blobs.100.10.npy')
# X_star = gradient_descent_algo(A)

eigenvalues = np.linalg.eigvals(X_optimal)
is_pd = np.all(eigenvalues > 0)
scores = np.einsum('...i,ij,...j->...', a, np.linalg.inv(X_optimal), a)

print(is_pd, np.all(scores <= 1.))

