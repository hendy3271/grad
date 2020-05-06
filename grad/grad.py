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

def differentiate(variable, source, gradient=1.):
    accumulation = 0.
    if not isinstance(variable, Variable):
        return 0.
    elif variable is source:
        return gradient
    elif variable.operation is None:
        return 0.
    elif not isinstance(variable.operation, str):
        return 0.
    
    operation = variable.operation

    if operation == 'func':
        args, gradients = variable.gradients
        iterable = zip(variable.parents, gradients)

        for parent, grd in iterable:
            if isinstance(parent, Variable):
                accumulation += gradient*differentiate(parent, source, grd(*args))
    else:
        operation = differential(operation)
        iterable = getattr(variable, operation)()

        for parent, grd in iterable:
            if isinstance(parent, Variable):
                accumulation += gradient*differentiate(parent, source, grd)
    
    return accumulation