from tester import tester
from grad import primitive
from math_wrap import sin, cos, sqrt, exp, log

@primitive([lambda *args: 2, lambda *args: -3])
def var2(x, y):
     return 2.*x - 3.*y

y = lambda x: x**2
x = 2.
tester(y, x, h=1.e-6, name='square')

y = lambda x: abs(x)
x = 2.
tester(y, x, h=1.e-6, name='abs')

y = lambda x: log(x)
x = 2.
tester(y, x, h=1.e-6, name='log')

y = lambda x: cos(x)
x = 2.
tester(y, x, h=1.e-6, name='cos')

y = lambda x: sin(x)
x = 1.
tester(y, x, h=1.e-6, name='sin')

y = lambda x: sqrt(x)
x = 2.
tester(y, x, h=1.e-6, name='sqrt')