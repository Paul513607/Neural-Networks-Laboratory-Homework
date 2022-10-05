import math

import numpy as np


def calculate_determinant(matrix):
    return np.linalg.det(matrix)


def get_transpose_matrix(matrix):
    return matrix.T


def find_minor_matrix_for_i_j(matrix, i, j):
    return np.array([list(row[:j]) + list(row[j + 1:]) for row in (list(matrix[:i]) + list(matrix[i + 1:]))])


def get_adjoint_matrix(matrix):
    minors_matrix = np.zeros((3, 3))
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[i])):
            temp_matrix = find_minor_matrix_for_i_j(matrix, i, j)
            minor_determinant = np.linalg.det(temp_matrix)
            minors_matrix[i, j] = minor_determinant * math.pow(-1, (i + j))
    return get_transpose_matrix(minors_matrix)


def get_inverse_matrix(matrix):
    determinant = calculate_determinant(matrix)
    if determinant == 0:
        raise Exception("The determinant of the matrix is 0!")
    inverse_determinant = 1 / determinant
    adjoint_matrix = get_adjoint_matrix(matrix)
    inverse_matrix = np.multiply(inverse_determinant, adjoint_matrix)
    return inverse_matrix


def solve(coefficient_matrix, results_matrix):
    inverse_matrix = get_inverse_matrix(coefficient_matrix)
    return inverse_matrix.dot(results_matrix)