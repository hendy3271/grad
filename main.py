from grad.math import sin, pi, atan2
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
        args=list(args)
        maximum=0.
        for i, arg in enumerate(args):
            if isinstance(arg, (list, tuple)):
                maximum = max(maximum, len(arg))
                args[i] = arg
            else:
                from itertools import cycle
                args[i] = cycle([arg])

        vector = []

        for vargs in zip(*args):
            vector.append(func(*vargs))
            if maximum == 0.:
                break
        else:
            return vector
        return vector[0]
        
    vfun.__vectorized__ = True
    return vfunc

@vectorize
def square(x):
    return x**2

@vectorize
def relu(x):
    if x < 0.:
        return 0.
    return x

sin, sin_ = vectorize(sin), sin

plot(x, sin(x), label = 'sin(x)')
plot(x, square(relu(x)), label = 'square(relu(x))')
legend()
show()