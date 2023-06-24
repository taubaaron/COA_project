import numpy as np
ESP = 10**(-6)


def calc_objective_func(x_vector, A, t):
    n = A.shape[1]
    L = x_vector.reshape((n, n))
    product_mat = L @ np.transpose(A)

    # val1: Calculate value of sigma {log(L_i_i)} i from 1 to n
    val1 = np.sum(np.log(np.diag(L)))

    # val2: Calculate value of -1/T * sigma {log(L_i_i - 10^-6)}  i from 1 to n
    val2 = np.sum(np.log(np.diag(L - ESP)))
    val2 = -(1/t) * val2

    # val3: Calculate the value of -1/T * sigma {log(1-norm(L*a_i)^2)} i from 1 to m
    norm_vector = np.linalg.norm(product_mat, axis=0)
    val3 = np.sum(np.log(1 - norm_vector ** 2))
    val3 = -(1/t) * val3

    return val1 + val2 + val3


def get_derivative_vector(x_vector, n):
    # Calculate the derivative of sigma {log(L_i_i)} i from 1 to n
    L = np.copy(x_vector.reshape((n, n)))
    L = np.diag(1/np.diag(L))
    return L.reshape((1, n**2))


def get_derivative_matrix(a_vector, n):
    # Calculate the directive of L*a_i (The i'th column of L * A^t)
    res = np.zeros((n, n ** 2))
    for i in range(n):
        res[i, i * n: (i + 1) * n] = a_vector[0:n]

    return res


def calc_gradient(x_vector, A, t):
    m = A.shape[0]
    n = A.shape[1]
    L = x_vector.reshape((n, n))
    product_mat = L @ np.transpose(A)

    # v1: Calculate the derivative of sigma {log(L_i_i)} i from 1 to n
    v1 = get_derivative_vector(x_vector, n)

    # v2: Calculate the derivative of -1/T * sigma {log(L_i_i - 10^-6)} i from 1 to n
    v2 = get_derivative_vector(x_vector - ESP, n)
    v2 = -(1/t) * v2

    # v3: Calculate the derivative of -1/T * sigma {log(1-norm(L*a_i)^2)} i from 1 to m
    v3 = np.zeros((1, n**2))

    for i in range(m):
        # Calculate the L*a_i (The i'th column of L * A^t)
        product_mat_col_i = product_mat[:, i]
        # Use chain rule the get the derivative of log(1-norm(L*a_i)^2)

        # 1. Calculate the derivative of log(1-norm(L*a_i)^2)
        der1 = -1 / (1 - np.linalg.norm(product_mat_col_i) ** 2)
        # 2. Calculate the derivative of norm(L*a_i)^2
        der2 = 2 * product_mat_col_i.reshape((1, n))
        # 3. Calculate the derivative of L*a_i
        der3 = get_derivative_matrix(product_mat_col_i, n)

        v3 = v3 + der1 * der2 @ der3

    v3 = -(1/t) * v3

    final_gradient = np.transpose(v1 + v2 + v3)
    return final_gradient


def backtracking(func, grad_val, x_vector):
    alpha = 0.5
    beta = 0.5
    t = 1/beta

    func_val = 1
    linear_approximation = 0
    while func_val > linear_approximation:
        t = beta * t
        func_val = func(x_vector - t * grad_val)
        linear_approximation = func(x_vector) - alpha * t * np.linalg.norm(grad_val) ** 2

    return t


def project_to_upper_triangular_matrix(x_vector, n):
    L = x_vector.reshape((n, n))
    grid = np.indices((n, n))
    L[grid[0] > grid[1]] = 0.0

    return L.reshape((1, n**2))


def solver(A: np.ndarray):
    n = A.shape[1]
    m = 10**(-6)
    t = 10

    # Initiate a random vector that represents an upper triangular matrix
    #x_vector = np.random.normal(0., 0.1, size=(1, n**2))
    x_vector = (1/100) * np.ones((1, n ** 2))
    x_vector = project_to_upper_triangular_matrix(x_vector, n)
    error = (1/(2*m)) * np.linalg.norm(calc_gradient(x_vector, A, t))**2

    # Perform gradient decent
    while error > 10**-3:
        grad_val = np.transpose(calc_gradient(x_vector, A, t))
        step_size = backtracking(lambda x: calc_objective_func(x, A, t), grad_val, x_vector)
        x_vector += x_vector - step_size * grad_val
        x_vector = project_to_upper_triangular_matrix(x_vector, n)
        error = (1 / (2 * m)) * np.linalg.norm(calc_gradient(x_vector, A, t)) ** 2
        #boolean += 1

    L = x_vector.reshape((n, n))
    PD_matrix = np.linalg.inv(np.transpose(L) @ L)

    return PD_matrix


if __name__ == '__main__':
    A_mat = np.arange(12).reshape((3, 4))
    PD_mat = solver(A_mat)
    print(PD_mat)
