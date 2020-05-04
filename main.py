from grad import Variable, trace

a = Variable(10)
b = Variable(2)
c = 5
d = c*a
e = d*b

trace(e, a)
