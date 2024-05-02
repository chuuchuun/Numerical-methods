import math
import time
from copy import deepcopy

import matplotlib.pyplot as plt

def jacobi(A, b, tolerance=10**-9, max_iterations=1000):
    n = len(A)
    x = [1 for _ in range(n)]
    residuals = []  # Store the residual norms
    residual_norm = 1
    last_residual_norm = math.inf
    iteration = 0
    while residual_norm > tolerance:
        iteration += 1
        x_new = [0 for _ in range(n)]
        for i in range(n):
            sum_Ax = sum(A[i][j] * x[j] for j in range(n) if i != j)
            x_new[i] = (b[i] - sum_Ax) / A[i][i]

        # Update residual vector and norm using the new approximation x_new
        Ax = [sum(A[i][j] * x_new[j] for j in range(n)) for i in range(n)]
        residual_vector = [Ax[i] - b[i] for i in range(n)]
        residual_norm = math.sqrt(sum(comp ** 2 for comp in residual_vector))
        residuals.append(residual_norm)

        if residual_norm < tolerance:
            return x_new, list(range(1, iteration + 1)), residuals
        if residual_norm > last_residual_norm:  # Check for divergence
            print("Divergence detected.")
            return x_new, list(range(1, iteration + 1)), residuals
        last_residual_norm = residual_norm
        x = x_new

    return x, list(range(1, max_iterations + 1)), residuals  # Ensure output even if not converged

def gauss_seidel(A, b, tolerance=10**-9, max_iterations=1000):
    n = len(A)
    x = [1 for _ in range(n)]
    residuals = []  # Store the residual norms
    residual_norm = 1
    last_residual_norm = math.inf
    iteration = 0
    while residual_norm > tolerance:
        iteration += 1
        for i in range(n):
            sum_Ax = sum(A[i][j] * x[j] for j in range(i))
            sum_Ax += sum(A[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (b[i] - sum_Ax) / A[i][i]

        # Update residual vector and norm using the current approximation x
        residual_vector = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
        residual_norm = math.sqrt(sum(comp ** 2 for comp in residual_vector))
        residuals.append(residual_norm)  # Append the current residual norm

        if residual_norm < tolerance:
            return x, list(range(1, iteration + 1)), residuals
        if residual_norm > last_residual_norm:  # Check for divergence
            print("Divergence detected.")
            return x, list(range(1, iteration + 1)), residuals
        last_residual_norm = residual_norm
    return x, list(range(1, max_iterations + 1)), residuals  # Ensure output even if not converged


def create(n, a1, a2, a3):
    A = [[0 for _ in range(n)] for _ in range(n)]
    for y in range(n):
        for x in range(n):
            if x == y:
                A[y][x] = a1
            elif x == y - 1 or x == y + 1:
                A[y][x] = a2
            elif x == y - 2 or x == y + 2:
                A[y][x] = a3

    b = [math.sin((i+1) * (6+1)) for i in range(n)]
    return A, b


def eye(N):
    identity_matrix = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        identity_matrix[i][i] = 1
    return identity_matrix

def zadanie_b():
    N = 922
    a1 = 5 + 7
    a2 = -1
    a3 = -1
    A, b = create(N, a1, a2, a3)
    time_jacobi_start = time.time()
    x_jacobi, iterations_jacobi, residual_jacobi = jacobi(A, b)
    time_jacobi = time.time() - time_jacobi_start

    time_gauss_start = time.time()
    x_gauss, iterations_gauss , residual_gauss= gauss_seidel(A, b)
    time_gauss = time.time() - time_gauss_start

    print(f"Jacobi method for N = {N}")
    print(f"iterations: {len(iterations_jacobi)}")
    print(f"residuum in the end: {residual_jacobi[-1]}")
    print(f"time: {time_jacobi}")

    print(f"Gauss-Seidel method for N = {N}")
    print(f"iterations: {len(iterations_gauss)}")
    print(f"residuum in the end: {residual_gauss[-1]}")
    print(f"time: {time_gauss}")

    plt.figure(figsize=(10, 6))
    plt.plot(iterations_jacobi, residual_jacobi, label='Jacobi')
    plt.plot(iterations_gauss, residual_gauss, label='Gauss-Seidel')  # Plot Gauss-Seidel residuals
    plt.yscale('log')
    plt.title('Comparison of Residual Norms by Iteration')
    plt.xlabel('Iteration Number')
    plt.ylabel('Residual Norm (log scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig('zadanie_b.png')
    plt.show()


def zadanie_c():
    N = 922
    a1 = 3
    a2 = -1
    a3 = -1
    A, b = create(N, a1, a2, a3)
    time_jacobi_start = time.time()
    x_jacobi, iterations_jacobi, residual_jacobi = jacobi(A, b)
    time_jacobi = time.time() - time_jacobi_start

    time_gauss_start = time.time()
    x_gauss, iterations_gauss, residual_gauss = gauss_seidel(A, b)
    time_gauss = time.time() - time_gauss_start

    print(f"Jacobi method for N = {N}")
    print(f"iterations: {len(iterations_jacobi)}")
    print(f"residuum in the end: {residual_jacobi[-1]}")
    print(f"time: {time_jacobi}")

    print(f"Gauss-Seidel method for N = {N}")
    print(f"iterations: {len(iterations_gauss)}")
    print(f"residuum in the end: {residual_gauss[-1]}")
    print(f"time: {time_gauss}")

    plt.figure(figsize=(10, 6))
    plt.plot(iterations_jacobi, residual_jacobi, label='Jacobi')
    plt.plot(iterations_gauss, residual_gauss, label='Gauss-Seidel')  # Plot Gauss-Seidel residuals
    plt.yscale('log')
    plt.title('Comparison of Residual Norms by Iteration')
    plt.xlabel('Iteration Number')
    plt.ylabel('Residual Norm (log scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig('zadanie_c.png')
    plt.show()


def forward_substitution(L, b):
    n = len(b)
    y = [0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    return y


def backward_substitution(U, y):
    n = len(y)
    x = [0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x


def calculate_residual(A, x, b):
    n = len(A)
    r = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    residual_norm = math.sqrt(sum(comp ** 2 for comp in r))
    return residual_norm


def lu(A, N):
    U = deepcopy(A)
    L = eye(N)
    for i in range(1, N):
        for j in range(i):
            L[i][j] = U[i][j] / U[j][j]
            for k in range(j, N):
                U[i][k] -= L[i][j] * U[j][k]
    return L, U

def zadanie_d():
    N = 922
    a1 = 3
    a2 = -1
    a3 = -1
    A, b = create(N, a1, a2, a3)
    time_lu_start = time.time()
    L, U = lu(A, N)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    residual_norm = calculate_residual(A, x, b)
    time_lu = time.time() - time_lu_start

    print(f"LU factorization method for N = {N}")
    print(f"Residual norm: {residual_norm}")
    print(f"Time: {time_lu}")


def zadanie_e():
    N =[100, 100, 100, 100,  3000]
    a1 = 5 + 7
    a2 = -1
    a3 = -1
    times_jacobi = []
    times_gauss = []
    times_lu = []
    for i in range(len(N)):
        A, b = create(N[i], a1, a2, a3)
        time_jacobi_start = time.time()
        x_jacobi, iterations_jacobi, residual_jacobi = jacobi(A, b)
        time_jacobi = time.time() - time_jacobi_start
        times_jacobi.append(time_jacobi)
        time_gauss_start = time.time()
        x_gauss, iterations_gauss, residual_gauss = gauss_seidel(A, b)
        time_gauss = time.time() - time_gauss_start
        times_gauss.append(time_gauss)
        time_lu_start = time.time()
        L, U = lu(A, N[i])
        y = forward_substitution(L, b)
        x = backward_substitution(U, y)
        residual_norm = calculate_residual(A, x, b)
        time_lu = time.time() - time_lu_start
        times_lu.append(time_lu)
        print(f"done: lu {time_lu} jacobi {time_jacobi} gauss-seidel {time_gauss}")

    plt.figure(figsize=(10, 6))
    plt.plot([N[i] for i in range(len(N))], times_jacobi, label='Jacobi')
    plt.plot([N[i] for i in range(len(N))], times_gauss, label='Gauss-Seidel')  # Plot Gauss-Seidel residuals
    plt.plot([N[i] for i in range(len(N))], times_lu, label='LU factorization')  # Plot Gauss-Seidel residuals
    #plt.yscale('log')
    plt.title('Comparison of times for different methods depending on N')
    plt.xlabel('N')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.savefig('zadanie_e_3.png')
    plt.show()


if __name__ == '__main__':
    zadanie_b()
    zadanie_c()
    zadanie_d()
    zadanie_e()