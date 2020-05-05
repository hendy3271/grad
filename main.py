from grad import grad
from grad.math import sin, tanh, exp, pi
from grad.vector import vectorize, linspace
from matplotlib.pyplot import plot, show, legend

x = list(linspace(-pi, pi, 100))

@vectorize
def square(x):
    return x**2

@vectorize
def relu(x):
    if x < 0.:
        return 0.
    return x

@vectorize
def mul(x, y):
    return x*y

@vectorize
def div(x, y):
    return x/y

@vectorize
def add(x, y):
    return x+y

@vectorize
def sub(x, y):
    return x-y

@vectorize
def neg(x):
    return -x

tanh, tanh_ = vectorize(tanh), tanh
exp, exp_ = vectorize(exp), exp
sin, sin_ = vectorize(sin), sin

f = lambda x: sin(x)
y = f(x)
dydx = grad(f)(x)
ddydxdx = grad(grad(f))(x)

plot(x, y, label = 'y')
plot(x, dydx, label = 'dydx')
plot(x, ddydxdx, label = 'ddydxdx')
legend()
show()

f = lambda x: square(x)
y = f(x)
dydx = grad(f)(x)
ddydxdx = grad(grad(f))(x)

plot(x, y, label = 'y')
plot(x, dydx, label = 'dydx')
plot(x, ddydxdx, label = 'ddydxdx')
legend()
show()