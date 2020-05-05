from .grad import grad
center_grad = lambda y, h: lambda x: (y(x+h) - y(x-h))/(2*h)
center_dgrad = lambda y, h: lambda x: (y(x+h) -2*y(x) + y(x-h))/(h*h)

def tester(y, x, h=1.e-6, name=None, n=1):
    if name:
        print(name)

    e = 1.e-7

    y_ = y
    for i in range(n):
        dydx_c = center_grad(y_, h)
        _dydx = dydx_c(x)

        dydx = grad(y)
        _, dydx_x = dydx(x)

        

        s = 'd' * (i+1) + 'y' + 'dx' * (i+1)
        error = float(abs(dydx_x-_dydx))
        print('{s} = {x}, {s} ~ {a} : error = {e}'.format(s=s, e=error, x=dydx_x, a=_dydx))
        assert error < e
        y = dydx
        y_ = lambda x: dydx(x)[1]