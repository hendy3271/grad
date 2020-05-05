from math import *
from grad import Variable, primitive, simple_primitive

min_, max_ = min, max

sin_ = sin
cos_ = cos
tan_ = tan

sin = simple_primitive(sin_, [lambda x: cos(x)])
cos = simple_primitive(cos_, [lambda x: -sin(x)])
tan = simple_primitive(cos_, [lambda x: 1/cos(x)**2])

exp_ = exp
exp = simple_primitive(exp_, [lambda x: exp(x)])

sqrt_ = sqrt
sqrt = simple_primitive(sqrt_, [lambda x: 0.5/sqrt(x)])

log_ = log
log = simple_primitive(log_, [lambda x, base=exp(1): 1/(log(base)*x), lambda x, base=exp(1): log(x)/(base*log(base)**2)])

asin_ = asin
acos_ = acos
atan_ = atan
atan2_ = atan2

asin = simple_primitive(asin_, [lambda x: 1/sqrt(1-x**2)])
acos = simple_primitive(acos_, [lambda x: -1/sqrt(1-x**2)])
atan = simple_primitive(atan_, [lambda x: 1/(1+x**2)])
atan2 = simple_primitive(atan2_, [lambda x, y: y/(y**2+x**2), lambda x, y: -x/(x**2+y**2)])

sinh_ = sinh
cosh_ = cosh
tanh_ = tanh

sinh = simple_primitive(sinh_, [lambda x: cosh(x)])
cosh = simple_primitive(cosh_, [lambda x: sinh(x)])
tanh = simple_primitive(tanh_, [lambda x: 1-tanh(x)**2])

asinh_ = asinh
acosh_ = acosh
atanh_ = atanh

asinh = simple_primitive(asinh_, [lambda x: 1/sqrt(x**2+1)])
acosh = simple_primitive(acosh_, [lambda x: 1/(sqrt(x-1)*sqrt(x+1))])
atanh = simple_primitive(atanh_, [lambda x: 1/(1-x**2)])
