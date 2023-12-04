import numpy as np

def f(o,p,q):
    o_mean = np.average(o)
    a = q-o_mean
    ss = np.sum(a**2)
    b = p-q
    press = np.sum(b**2)
    f = 1 - press/ss
    return f
