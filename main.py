import test
from grad.math import sin, tanh, exp, pi
from grad.vector import vectorize, linspace
from matplotlib.pyplot import plot, show, legend

from grad import grad, Variable, derivatives

f = lambda x, y=2., z=3.: x**2 + 3*y - 2*z
x = Variable(0.3)
y = Variable(0.2)

z = f(x, y)

print(derivatives(z, [x, y]))
df_dxy = grad(f, argnum=[0, 1, 'z'])
print(df_dxy(0.3, 0.2, z=3.))

from grad import elementwise_grad as grad

x = list(linspace(-pi, pi, 100))

relu = vectorize(lambda x: max(x, 0.1*x))

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

@vectorize
def f1(x):
    y = 0.
    m = 1.
    for n in range(10):
        m *= n if n > 0 else 1
        y += x**n/m
    return y

y = f1(x)
dydx = grad(f1)(x)
ddydxdx = grad(grad(f1))(x)

plot(x, y, label = 'y')
plot(x, dydx, label = 'dydx')
plot(x, ddydxdx, label = 'ddydxdx')
legend()
show()

y = relu(x)
dydx = grad(relu)(x)
ddydxdx = grad(grad(relu))(x)

plot(x, y, label = 'y')
plot(x, dydx, label = 'dydx')
plot(x, ddydxdx, label = 'ddydxdx')
legend()
show()