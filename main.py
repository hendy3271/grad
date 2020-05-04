from grad import Variable, trace

p = lambda a, b: print('a  = {a}, b  = {b}'.format(a=a, b=b))
p_ = lambda a, b: print('a_ = {a}, b_ = {b}'.format(a=a, b=b))

a = 10.
a_ = Variable(10)
b = a
b_ = a_

p(a, b)
p_(a_, b_)
assert a == a_, b == b_

a = a + 1
a_ = a_ + 1

p(a, b)
p_(a_, b_)
assert a == a_, b == b_

a += 1.
a_ += 1.

p(a, b)
p_(a_, b_)
assert a == a_, b == b_

b = a*5
b_ = a_*5

p(a, b)
p_(a_, b_)
assert a == a_, b == b_

trace(a_)