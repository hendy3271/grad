from grad.math import sin, pi, atan2
from grad.vector import vectorize, linspace
from matplotlib.pyplot import plot, show, legend

x = list(linspace(-pi, pi))

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