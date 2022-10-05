import math


def calculate_2x2_matrix_determinant(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def calculate_3x3_matrix_determinant(matrix):
    positive_term1 = matrix[0][0] * matrix[1][1] * matrix[2][2]
    positive_term2 = matrix[0][2] * matrix[1][0] * matrix[2][1]
    positive_term3 = matrix[0][1] * matrix[1][2] * matrix[2][0]

    negative_term1 = matrix[0][2] * matrix[1][1] * matrix[2][0]
    negative_term2 = matrix[0][1] * matrix[1][0] * matrix[2][2]
    negative_term3 = matrix[0][0] * matrix[1][2] * matrix[2][1]

    return positive_term1 + positive_term2 + positive_term3 - negative_term1 - negative_term2 - negative_term3


def get_transpose_matrix(matrix):
    matrix_T = []
    for j in range(0, len(matrix)):
        curr_row = []
        for i in range(0, len(matrix)):
            curr_row.append(matrix[i][j])
        matrix_T.append(curr_row)

    return matrix_T


def find_minor_matrix_for_i_j(matrix, i, j):
    return [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]


# get A*
def get_adjoint_matrix(matrix):
    minors_matrix = []
    for i in range(0, len(matrix)):
        temp_arr = []
        for j in range(0, len(matrix[i])):
            temp_matrix = find_minor_matrix_for_i_j(matrix, i, j)
            minor_determinant = calculate_2x2_matrix_determinant(temp_matrix)
            temp_arr.append(minor_determinant * math.pow(-1, (i + j)))
        minors_matrix.append(temp_arr)
    return get_transpose_matrix(minors_matrix)


def get_inverse_matrix(matrix):
    determinant = calculate_3x3_matrix_determinant(matrix)
    if determinant == 0:
        raise Exception("The determinant of the matrix is 0!")
    adjoint_matrix = get_adjoint_matrix(matrix)
    inverse_determinant = 1 / determinant

    inverse_matrix = []
    for i in range(0, len(adjoint_matrix)):
        inverse_matrix.append([])
        inverse_matrix[i] = [x * inverse_determinant for x in adjoint_matrix[i]]
    return inverse_matrix


def solve(coefficient_matrix, results_matrix):
    inverse_matrix = get_inverse_matrix(coefficient_matrix)
    solution = []
    for i in range(0, len(inverse_matrix)):
        line_value = 0
        for j in range(0, len(inverse_matrix[i])):
            line_value += inverse_matrix[i][j] * results_matrix[j][0]
        solution.append(line_value)
    return solution
