from math import sin, cos, sqrt, exp, log
from grad import Variable, primitive, simple_primitive

min_, max_ = min, max

sin_ = sin
cos_ = cos

sin = simple_primitive(sin_, [lambda x: cos(x)])
cos = simple_primitive(cos_, [lambda x: -sin(x)])

exp_ = exp
exp = simple_primitive(exp_, [exp])

sqrt_ = sqrt
sqrt = simple_primitive(sqrt_, [lambda x: 0.5/sqrt(x)])

log_ = log
log = simple_primitive(log_, [lambda x, base=exp(1): 1/(log(base)*x), lambda x, base=exp(1): log(x)/(base*log(base)**2)])