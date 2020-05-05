from math import sin, pi
from matplotlib.pyplot import plot, show

def linspace(start, stop, N = 50):
    step = (stop-start)/(N-1)
    for n in range(N):
        yield start + n*step

x = list(linspace(-pi, pi))

def vectorize(func):
    def vfunc(*args):
        vector = []
        for vargs in zip(*args):
            vector.append(func(*vargs))
        return vector
    return vfunc

vsin = vectorize(sin)
plot(x, vsin(x))
show()