from grad import Variable, trace

a = Variable(10)
b = Variable(2)

c = a*b

c *= 5

d = a * 5

e = d*c
f = c*d

trace(e)
trace(f)