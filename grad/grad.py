from .variables import Variable

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
        return differentiate(y, x)
    return dfunc

def primitive(gradients):
    def wrapper(func):
        def wrapped_func(*args):
            return Variable(func(*args), parents=args, gradients=gradients) if len(args) > 0 else func(*args) 
        return wrapped_func
    return wrapper

simple_primitive = lambda func, gradients: primitive(gradients)(func)

def differentiate(y, x):
    if y is x:
        # if I am x then dy/ds is actually dy/dx
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
            dy_dx += dy_ds(*y.parents) * differentiate(s, x)
    
    return dy_dx