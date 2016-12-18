from scipy.optimize import minimize, rosen, rosen_der
import numpy as np
def func(z):
    x =  z[0:2]
    y = z[2:]
    return x[0]**2 + x[1]**2 + y[0]**2 + (y[1]+1) ** 2
def diff_func(z):
    x =  z[0:2]
    y = z[2:]
    return np.array([2 * x[0], 2 * x[1], 2 * y[0], 2 * y[1] + 2])
if __name__ == '__main__':
    x0 = np.array([(1.0, 1.0), (1.0, 2.0)])
    print(x0)
    res = minimize(func, x0, jac=diff_func, method='L-BFGS-B')
    print(res)
    res = minimize(func, x0, method='L-BFGS-B')
    print(res)
