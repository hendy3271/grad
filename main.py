from grad import grad

y = lambda x: 1/(x*x*x)
dydx = grad(y)

x = 3
h = 0.000001
y_12, dydx_12 = dydx(x)

a_dydx = (y(x+h) - y(x-h))/(2*h)

print('y({x}) = {y}, dydx({x}) = {dydx}'.format(x=x, y=y_12, dydx=dydx_12))
print('dydx ~ {}'.format(a_dydx))