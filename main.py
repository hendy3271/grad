from math import sin, pi, atan2
from matplotlib.pyplot import plot, show, legend

def linspace(start, stop, N = 50):
    step = (stop-start)/(N-1)
    for n in range(N):
        yield start + n*step

x = list(linspace(-pi, pi))

# need grad_vectorise
# and vectorize grad
def vectorize(func):
    def vfunc(*args):
        maximum = 0.
        for arg in args:
            if isinstance(arg, (list, tuple)):
                maximum = max(maximum, len(arg))

        if maximum == 0.:
            return func(*args)

        sargs = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                sargs.append(list(arg))
            else:
                from itertools import cycle
                sargs.append(cycle([arg]))

        vector = []

        for vargs in zip(*sargs):
            vector.append(func(*vargs))
        return vector
    return vfunc

@vectorize
def square(x):
    return x**2

@vectorize
def relu(x):
    if x < 0.:
        return 0.
    return x

vsin = vectorize(sin)

plot(x, vsin(x), label = 'sin(x)')
plot(x, square(relu(x)), label = 'square(relu(x))')
legend()
show()