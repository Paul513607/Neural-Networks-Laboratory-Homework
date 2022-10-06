import numpy as np
import numpy.linalg

import parser
import py_solver
import numpy_solver as np_solver


if __name__ == '__main__':
    coefficient_matrix, results_matrix = parser.get_equation_system_from_file("./equations.txt")
    try:
        print("Solution obtained using vanilla python: ", py_solver.solve(coefficient_matrix, results_matrix))
    except Exception as e:
        print(e)

    coefficient_matrix_np = np.array(coefficient_matrix)
    results_matrix_np = np.array(results_matrix)
    try:
        print("Solution using numpy calculations: ", np_solver.solve(coefficient_matrix_np, results_matrix_np))
    except Exception as e:
        print(e)

    try:
        print("Solution using numpy equation system solver: ", numpy.linalg.solve(coefficient_matrix_np, results_matrix_np))
    except Exception as e:
        print(e)
