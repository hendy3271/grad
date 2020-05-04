from grad import Variable, trace, grad

a = Variable(12)
b = Variable(3)
c = 5
d = c*a

e = a*a+2*a+4

y = lambda x: x*x+2*x+4
dydx = grad(y)

grd = trace(e, a)
print(e)
print(grd)

print(dydx(12))