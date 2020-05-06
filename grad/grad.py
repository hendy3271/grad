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
            if len(args) > 0:
                return Variable(func(*args), parents=args, gradients=gradients)
            return func(*args)
        return wrapped_func
    return wrapper

def simple_primitive(func, gradients):
    return primitive(gradients)(func)

def differentiate(y, x, ds_dy=1.):
    if y is x:
        return ds_dy
    elif not isinstance(y, Variable) or y.parents is None or y.gradients is None:
        return 0.
    
    dy_dx = 0.

    # Note s is the intermim variable
    for s, ds_da in zip(y.parents, y.gradients):
        if isinstance(s, Variable):
            # dy/dx = dy/ds*ds/dx
            dy_dx += ds_dy*differentiate(s, x, ds_da(*y.parents))
    
    return dy_dx