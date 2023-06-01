from sympy import *
from IPython.display import display, Latex

def display_result(a, b=None, c=None):
  if c is None:
    if b is None:
      res = "$${}$$".format(latex(a, mat_delim='('))
    else:
      res = "$${} = {}$$".format(latex(a, mat_delim='('), latex(b, mat_delim='('))
  else:
    res = "$${} = {} = {}$$".format(latex(a, mat_delim='('), latex(b, mat_delim='('), latex(c, mat_delim='('))
  display(Latex(res))

def display_latex_result(a, b=None):
  if b is None:
    res = "$${}$$".format(a)
  else:
    res = "$${} = {}$$".format(a, latex(b, mat_delim='('))
  display(Latex(res))
