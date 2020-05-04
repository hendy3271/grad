from grad import Variable, trace

a = Variable(12)
b = Variable(3)
c = 5
d = c*a

e = a+2+a+a+100

grd = trace(e, a)
print(grd)