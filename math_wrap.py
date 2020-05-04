from math import sin, cos, sqrt, exp
from grad import Variable, grad_wrap

min_, max_ = min, max

sin_ = sin
cos_ = cos

sin = grad_wrap(sin_, cos_)
cos = grad_wrap(cos_, lambda x: -sin_(x))

exp_ = exp
exp = grad_wrap(exp_, exp_)

sqrt_ = sqrt
sqrt = grad_wrap(sqrt_, lambda x: 0.5/sqrt_(x))