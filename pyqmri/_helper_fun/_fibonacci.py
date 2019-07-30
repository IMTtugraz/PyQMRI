
import numpy as np


def run(num):
    if num < 3:
        if num < 2:
            fib = 1
        else:
            np.zeros((2))
            fib[0] = 1
            fib[1] = 1
    else:
        fib = np.zeros(num)
        fib[0] = 1
        fib[1] = 1
        for ii in range(2, num):
            fib[ii] = fib[ii - 2] + fib[ii - 1]
    return fib
