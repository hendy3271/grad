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

def primitive(grads):
    def wrapper(func):
        def wrapped_func(*args):
            value = func(*args)
            gradients = []
            parents = []
            
            for arg, gradient in zip(args, grads):
                if isinstance(arg, Variable):
                    parents.append(arg)
                    gradients.append(gradient)

            if len(parents) > 0 and len(args) > 0:
                return Variable(value, parents=parents, operation='func', gradients=(args, gradients))
            return value
        return wrapped_func
    return wrapper

def simple_primitive(func, grads):
    return primitive(grads)(func)

differential = lambda operation: operation[:2] + 'd' + operation[2:]

def differentiate(y, x, ds_dy=1.):
    dy_dx = 0.
    if not isinstance(y, Variable):
        return 0.
    elif y is x:
        return ds_dy
    elif y.parents is None:
        return 0.

    # Note s is the intermim variable
    args, gradients = y.gradients

    for s, ds_da in zip(y.parents, gradients):
        if isinstance(s, Variable):
            # dy/dx = dy/ds*ds/dx
            dy_dx += ds_dy*differentiate(s, x, ds_da(*args))
    
    return dy_dx