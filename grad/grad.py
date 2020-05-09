from .variables import Variable
from .vector import vectorize

def grad(func, argnum=0):
    def dfunc(*args, **kwargs):
        if isinstance(argnum, (int, float)):
            # trace args[argnum]
            args=list(args)
            args[argnum] = x = Variable(args[argnum])
        else:
            raise TypeError

        # Foward pass
        y = func(*args, **kwargs)

        # Reverse pass
        return derivatives(y, x)
    return dfunc

elementwise_grad = lambda f: vectorize(grad(f))

def primitive(gradients):
    def wrapper(func):
        def wrapped_func(*args):
            return Variable(func(*args), parents=args, gradients=gradients) if len(args) > 0 else func(*args) 
        return wrapped_func
    return wrapper

simple_primitive = lambda func, gradients: primitive(gradients)(func)

def derivative(y, x):
    if y is x:
        # If I am x then asking for dy/dx is actually dx/dx = 1.0
        return 1.
    elif not isinstance(y, Variable):
        return 0.
    elif y.parents is None or y.gradients is None:
        return 0.
    
    dy_dx = 0.

    # Note s is the intermim variable/current parent
    for s, dy_ds in zip(y.parents, y.gradients):
        if isinstance(s, Variable):
            # dy/dx = dy/ds*ds/dx
            dy_dx += dy_ds(*y.parents) * derivative(s, x)
    
    return dy_dx

def derivatives(y, xs):
    if isinstance(xs, Variable):
        return derivative(y, xs)
    elif len(xs) == 0:
        return 0.
    elif len(xs) == 1:
        return derivative(y, xs[0])
    else:
        n = len(xs)
    
    dy_dxs = [0.]*n

    if y in xs:
        for i, x in enumerate(xs):
            if y is x:
                dy_dxs[i] = 1.
    
    if not isinstance(y, Variable):
        return dy_dxs
    elif y.parents is None or y.gradients is None:
        return dy_dxs

    # Note s is the intermim variable/current parent
    for s, dy_ds in zip(y.parents, y.gradients):
        if isinstance(s, Variable):
            # dy/dx = dy/ds*ds/dx
            dy_ds_ = dy_ds(*y.parents)
            for i, ds_dx in enumerate(derivatives(s, xs)):
                dy_dxs[i] += dy_ds_ * ds_dx
    
    return dy_dxs