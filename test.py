from grad import primitive, tester
from grad.math import *

@primitive([lambda x, y: 2, lambda x, y: -3])
def var2(x, y):
     return 2.*x - 3.*y

N = 3
x = 2.
x_2 = 0.5

y = lambda x: x**2
tester(y, x, h=1.e-6, name='square', n=N)

y = lambda x: abs(x)
tester(y, x, h=1.e-6, name='abs', n=N)

y = lambda x: cos(x)
tester(y, x, h=1.e-6, name='cos', n=N)

y = lambda x: sin(x)
tester(y, x, h=1.e-6, name='sin', n=N)

y = lambda x: sqrt(x)
tester(y, x, h=1.e-6, name='sqrt', n=N)

y = lambda x: log(x)
tester(y, x, h=1.e-6, name='log', n=N)

y = lambda x: exp(x)
tester(y, x, h=1.e-6, name='exp', n=N)

y = lambda x: asin(x)
tester(y, x_2, h=1.e-6, name='asin', n=N)

y = lambda x: acos(x)
tester(y, x_2, h=1.e-6, name='acos', n=N)

y = lambda x: atan(x)
tester(y, x, h=1.e-6, name='atan', n=N)

y = lambda x: atan2(x, 1)
tester(y, x, h=1.e-6, name='atan2_0', n=N)

y = lambda x: atan2(1, x)
tester(y, x, h=1.e-6, name='atan2_1', n=N)

y = lambda x: atan2(x, x)
tester(y, x, h=1.e-6, name='atan2_2', n=N)

y = lambda x: sinh(x)
tester(y, x, h=1.e-6, name='sinh', n=N)

y = lambda x: cosh(x)
tester(y, x, h=1.e-6, name='cosh', n=N)

y = lambda x: tanh(x)
tester(y, x, h=1.e-6, name='tanh', n=N)

y = lambda x: asinh(x)
tester(y, x, h=1.e-6, name='asinh', n=N)

y = lambda x: acosh(x)
tester(y, x, h=1.e-6, name='acosh', n=N)

y = lambda x: atanh(x)
tester(y, x_2, h=1.e-6, name='atanh', n=N)

y = lambda x: var2(x, x)
tester(y, x, h=1.e-6, name='user primitive', n=N)