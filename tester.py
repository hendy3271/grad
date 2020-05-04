from grad import grad, primitive, Variable
center_grad = lambda y, h: lambda x: (y(x+h) - y(x-h))/(2*h)
center_dgrad = lambda y, h: lambda x: (y(x+h) -2*y(x) + y(x-h))/(h*h)

def tester(y, x, h=1.e-6, name=None):
    if name:
        print(name)

    dydx = grad(y)

    dydx_cg = center_grad(y, h)

    y_x, dydx_x = dydx(x)

    _y = y(x)
    _dydx = dydx_cg(x)

    print('y = {}, dydx = {}'.format(y_x, dydx_x))
    print('y ~ {}, dydx ~ {}'.format(_y, _dydx))
    error = abs(y_x-_y)
    derror = abs(dydx_x-_dydx)
    e = 1.e-6
    assert error < e
    assert derror < e 

def tester2(y, x, h=1.e-11, name=None):
    if name:
        print(name)

    dydx = grad(y)
    ddydxdx = grad(dydx)

    dydx_cg = center_grad(y, h)
    ddydxdx_cg = center_dgrad(y, h)

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
    assert dderror < e*10
    assert abs(dydx_x_ - dydx_x) < e