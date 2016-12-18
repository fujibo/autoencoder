from scipy.optimize import minimize

def func(x):
    return x**4 + 5*x + 6
def diff_func(x):
    return 4 * x**3 + 5
if __name__ == '__main__':
    res = minimize(fun=func, x0=0, jac=diff_func, method='L-BFGS-B')
    print(res)
