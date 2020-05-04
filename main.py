from grad import grad

y = lambda x: (x**2 + 5*x - 2)/(3**x)
dydx = grad(y)

h = 1.e-7
dydx_ca = lambda x: (y(x+h) - y(x-h))/(2*h)

x = 3

y_12, dydx_12 = dydx(x)

a_dydx = dydx_ca(x)

print('y({x}) = {y}, dydx({x}) = {dydx}'.format(x=x, y=y_12, dydx=dydx_12))
print('dydx ~ {}'.format(a_dydx))
error = abs(dydx_12-a_dydx)
assert error < h