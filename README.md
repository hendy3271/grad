# Grad
Automatic differentiation for python floats with some vectorization to make plotting nicer.

## The idea
This is just for learning about automatic differentiation and draws inspiration from [autograd](https://github.com/HIPS/autograd) (python) and [autodiff](https://github.com/autodiff/autodiff) (C++) projects so that I could better understand what is happening under the hood.

I have redefined floats as an object called Variable this makes sure that the original float methods work as normal and allows the user to use a Variable as if it were float in normal operations then if so desired trace back to a derivative along the way. To do this a few parameters need to be set when an operation creates a new Variable, lets follow along with some code:

```python
from grad import Variable, differentiate

a = Variable(1.)
print(a.parents)
print(a.gradients)
>>> None # this is the start of a trace, so we can't go further back
>>> None
```
Counter intuitively a Variable (like a float) is immutable (not variable) but by creating a variable we are marking that we wish to trace our operations. If we pass a float into the creation of a variable then the variable has no parents, operations or gradients. When the Variable is created by a primitive method or primitive function then the parents are the arguments used in that method (including self) or function.

```python
b = 2*a
print(b.parents)
print(b.gradients)
>>> [1.0, 2.0] # note that 2.0 is actually the variable named 'a', 1.0 is a float
>>> [lambda a, b: gradient, ...] # these functions return the gradient when passed parents
```

So now we have registered that `b` has come into existance from its parents (which were arguements) and that the gradient based on each parent is defined by the functions (whose input is the parent). So we have a link from `b` back to `a`.

```python
print(differentiate(b, a)) # find db/da
>> 2.0
```

When differentiate is called it aims to find the 'db/da' which means that it starts at b and traces backwards until it finds `a` (note: there can be many paths back to `a` this is handled by recursion!).

```python
def differentiate(y, x, dy_ds=1.):
    if y is x:
        # if I am x then dy/ds is actually dy/dx
        return dy_ds
    elif not isinstance(y, Variable) or y.parents is None or y.gradients is None:
        return 0.
    
    dy_dx = 0.

    # Note s is the intermim variable/current parent
    for s, ds_da in zip(y.parents, y.gradients):
        if isinstance(s, Variable):
            # dy/dx = dy/ds*ds/dx = dy/ds*(ds/da*da/dx)
            dy_dx += dy_ds*differentiate(s, x, ds_da(*y.parents))
    
    return dy_dx
```

Now differentiate will first check if `y` is at the final value in which case it returns the derivative. Otherwise, the function check that value recieved is traceable (ie a Variable). If the current value is either non-traceable or has no parents then the derivative is 0 (note: this could be in one of the recursion branches) so 0 is returned.

Otherwise, for each of the parents the derivative of the parent is requested and mutliplied by the current locations derivative. For example,

`a = 1.`
`b = 2*a`

tracing this back from `b` we get two parents `[2, 1]`