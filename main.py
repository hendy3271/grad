from grad import grad

y = lambda x: x*x+2*x+4
dydx = grad(y)

x = 3
y_12, dydx_12 = dydx(x)
print('y({x}) = {y}, dydx({x}) = {dydx}'.format(x=x, y=y_12, dydx=dydx_12))