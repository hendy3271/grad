from grad import grad, primitive, Variable
from math_wrap import sin, cos, sqrt, exp

@primitive([lambda *args: 2, lambda *args: -3])
def var2(x, y):
     return 2.*x - 3.*y

y = lambda x: x**2
dydx = grad(y)
ddydxdx = grad(dydx)

h = 1.e-6
center_grad = lambda y, h: lambda x: (y(x+h) - y(x-h))/(2*h)
dydx_cg = center_grad(y, h)
ddydxdx_cg = center_grad(dydx_cg, h)

x = 2.

y_x, dydx_x = dydx(x)
dydx_x_, ddydxdx_x = ddydxdx(x)

_y = y(x)
_dydx = dydx_cg(x)
_ddydxdx = ddydxdx_cg(x)

print('y = {}, dydx = {}, ddydxdx = {}'.format(y_x, dydx_x, ddydxdx_x))
print('y ~ {}, dydx ~ {}, ddydxdx ~ {}'.format(_y, _dydx, _ddydxdx))
error = abs(y_x-_y)
derror = abs(dydx_x-_dydx)
dderror = abs(ddydxdx_x-_ddydxdx)
e = 1.e-3
assert error < e
assert derror < e 
assert dderror < e
assert abs(dydx_x_ - dydx_x) < e