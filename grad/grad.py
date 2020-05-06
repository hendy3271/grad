from .variables import Variable

def grad(func, argnum=0):
    def dfunc(*args, **kwargs):
        if isinstance(argnum, (int, float)):
            # wrap args[arg]
            args=list(args)
            var = args[argnum]
            vectorized = False
            if isinstance(var, (list, tuple)):
                vectorized = True
                for i, v in enumerate(var):
                    var[i] = Variable(v)
            if isinstance(var, float):
                var = Variable(args[argnum])
            args[argnum] = var
            pass 
        else:
            raise TypeError

        # Foward pass
        value = func(*args, **kwargs)

        # Reverse pass
        if vectorized:
            gradient = []
            for vr, vl in zip(var, value):
                gradient.append(differentiate(vl, vr))
        else:
            gradient = differentiate(value, var)
        return gradient
    return dfunc

def primitive(gradients):
    def wrapper(func):
        def wrapped_func(*args):
            return Variable(func(*args), parents=args, gradients=gradients) if len(args) > 0 else func(*args) 
        return wrapped_func
    return wrapper

def simple_primitive(func, gradients):
    return primitive(gradients)(func)

def differentiate(y, x, dy_ds=1.):
    if y is x:
        # if I am x then dy/ds is actually dy/dx
        return dy_ds
    elif not isinstance(y, Variable):
        return 0.
    elif y.parents is None or y.gradients is None:
        return 0.
    
    dy_dx = 0.

    # Note s is the intermim variable/current parent
    for s, ds_da in zip(y.parents, y.gradients):
        if isinstance(s, Variable):
            # dy/dx = dy/ds*ds/dx = dy/ds*(ds/da*da/dx)
            dy_dx += dy_ds*differentiate(s, x, ds_da(*y.parents))
    
    return dy_dx