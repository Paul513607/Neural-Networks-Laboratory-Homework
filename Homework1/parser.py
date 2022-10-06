import numpy as np


def get_data_from_file(filepath):
    file = open(filepath, "r")
    equation1_str = file.readline()
    equation2_str = file.readline()
    equation3_str = file.readline()
    file.close()

    # Replace the multiplication operator and the newline character with empty string
    equation1_str = equation1_str.replace("*", "").replace("\n", "")
    equation2_str = equation2_str.replace("*", "").replace("\n", "")
    equation3_str = equation3_str.replace("*", "").replace("\n", "")
    return equation1_str, equation2_str, equation3_str


def find_result_matrix(equation1_remainder, equation2_remainder, equation3_remainder):
    equation1_remainder.replace("=", "")
    equation2_remainder.replace("=", "")
    equation3_remainder.replace("=", "")

    result_matrix = [int(equation1_remainder)], [int(equation2_remainder)], [int(equation3_remainder)]
    return result_matrix


# find the coefficients of the current equation
def find_coefficient_line(equation_str):
    line_arr = []

    # find x
    index = equation_str.find('x')
    if index == -1:
        line_arr.append(0)
        # Edge case if the is no x and no sign before the y coefficient
        if equation_str[0] not in ('+', '-'):
            equation_str = '+' + equation_str
    else:
        if index == 0:
            line_arr.append(1)
        elif index == 1:
            line_arr.append(int(equation_str[0]))
        else:
            line_arr.append(int(equation_str[0:index]))
        equation_str = equation_str[index + 1:]

    # find y
    index = equation_str.find('y')
    if index == -1:
        line_arr.append(0)
    # the index for y in the current string won't be 0 because it always has a +/- sign before it
    else:
        if index == 1:
            if equation_str[0] == '+':
                line_arr.append(1)
            elif equation_str[0] == '-':
                line_arr.append(-1)
        else:
            line_arr.append(int(equation_str[0:index]))
        equation_str = equation_str[index + 1:]

    # find z
    index = equation_str.find('z')
    if index == -1:
        line_arr.append(0)
    # the index for z in the current string won't be 0 because it always has a +/- sign before it
    elif index == 1:
        if equation_str[0] == '+':
            line_arr.append(1)
        elif equation_str[0] == '-':
            line_arr.append(-1)
    else:
        line_arr.append(int(equation_str[0:index]))

    return line_arr


def get_equation_system_from_file(filepath):
    equation1_str, equation2_str, equation3_str = get_data_from_file(filepath)
    equation1_str = equation1_str.lower()
    equation2_str = equation2_str.lower()
    equation3_str = equation3_str.lower()

    results_matrix = find_result_matrix(equation1_str.split("=")[1], equation2_str.split("=")[1],
                                        equation3_str.split("=")[1])
    equation1_str = equation1_str.split("=")[0]
    equation2_str = equation2_str.split("=")[0]
    equation3_str = equation3_str.split("=")[0]

    coefficient_matrix = [find_coefficient_line(equation1_str), find_coefficient_line(equation2_str),
                          find_coefficient_line(equation3_str)]

    print(coefficient_matrix, results_matrix)
    return coefficient_matrix, results_matrix
