from math import sin, cos, sqrt, exp
from grad import Variable, primitive, simple_primitive

min_, max_ = min, max

sin_ = sin
cos_ = cos

sin = simple_primitive(sin_, [cos])
cos = simple_primitive(cos_, [lambda x: -sin(x)])

exp_ = exp
exp = simple_primitive(exp_, [exp])

sqrt_ = sqrt
sqrt = simple_primitive(sqrt_, [lambda x: 0.5/sqrt(x)])