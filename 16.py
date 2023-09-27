import numpy as np


def inv_gauss_jordan(A):
    n = len(A)

    if A.shape[0] != A.shape[1]:
        return None

    # Crear la matriz aumentada [A|I], donde I es la matriz identidad
    augmented_matrix = np.hstack([A, np.eye(n)])

    # Aplicar eliminación de Gauss-Jordan
    for col in range(n):
        # Hacer el elemento diagonal 1
        augmented_matrix[col] /= augmented_matrix[col, col]

        # Hacer los demás elementos de la columna 0
        for row in range(n):
            if row != col:
                augmented_matrix[row] -= (
                    augmented_matrix[row, col] * augmented_matrix[col]
                )

    # La matriz inversa es la parte derecha de la matriz aumentada
    inverse_matrix = augmented_matrix[:, n:]

    return inverse_matrix


A = np.array([[1, 2], [3, 4]])

print(
    "a) The condition number of A in the infinite-norm is ",
    np.round(
        np.linalg.norm(A, np.inf) * np.linalg.norm(inv_gauss_jordan(A), np.inf), 14
    ),
)

b = np.array([[1], [1]])
x = np.linalg.solve(A, b)

eigenvalues, eigenvectors = np.linalg.eig(A @ A.T)
dominant_eigenvector = eigenvectors[:, np.argmax(np.abs(eigenvalues))]

epsilon = 1.4914e-16

delta_be = epsilon * dominant_eigenvector

delta_b = np.random.rand(2, 1) * 1e-14

relative_error_b = np.linalg.norm(delta_be, np.inf) / np.linalg.norm(b, np.inf)
x_perturbed = np.linalg.solve(A, b + delta_be)
delta_x = x_perturbed - x
relative_error_x = np.linalg.norm(delta_x, np.inf) / np.linalg.norm(x, np.inf)

print(relative_error_x / relative_error_b)

epsilon = 1.621123e-15
delta_A = epsilon * np.outer(dominant_eigenvector, dominant_eigenvector)


relative_error_A = np.linalg.norm(delta_A, np.inf) / np.linalg.norm(A, np.inf)
x_perturbed = np.linalg.solve(A + delta_A, b)
delta_x = x_perturbed - x
relative_error_x = np.linalg.norm(delta_x, np.inf) / np.linalg.norm(x, np.inf)

print(relative_error_x / relative_error_A)
