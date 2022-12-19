import sympy as sp  # https://problemsolvingwithpython.com/10-Symbolic-Math/10.06-Solving-Equations/

def solver(equation):
    try:
        if "=" in equation:
            sympy_eq = sp.sympify("Eq(" + equation.replace("=", ",") + ")") # https://stackoverflow.com/a/50047781
            return f"$$ = {sp.solve(sympy_eq)} $$"
        else:
            return f"$$ = {eval(equation)} $$" # type: ignore
    except:
        return "The input doesn't seem to be correct, try typing out the equation"


# Example
# print(solver("2*x+6=12"))
