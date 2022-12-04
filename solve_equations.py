import sympy as sp

def solver(equation):
    if "=" in equation:
        sympy_eq = sp.sympify("Eq(" + equation.replace("=", ",") + ")") # https://stackoverflow.com/questions/50043189/solve-equation-string-with-python-to-every-symbol/50047781#50047781
        return sp.solve(sympy_eq)
    else:
        return eval(equation)

# Example
print(solver("2*x+6=12"))