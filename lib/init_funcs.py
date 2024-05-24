import numpy as np

def sinxy(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def constxy(x=0, y=0):
    return 1

def avgxy(x, y):
    return (x+y)/2