from .variables import Variable

def grad(func, arg=0):
    def dfunc(*args, **kwargs):
        if isinstance(arg, (int, float)):
            # wrap args[arg]
            args=list(args)
            var = Variable(args[arg])
            args[arg] = var
            pass 
        elif isinstance(arg, str):
            # wrap kwargs[arg]
            var = Variable(args[arg])
            kwargs[arg] = var
            pass
        else:
            raise TypeError

        # Foward pass
        if getattr(func, '__isgrad__', False):
            _, value = func(*args, **kwargs)
        else:
            value = func(*args, **kwargs)

        # Reverse pass
        gradient = differentiate(value, var)

        return value, gradient
    dfunc.__isgrad__ = True
    return dfunc

def primitive(grads):
    def wrapper(func):
        def wrapped_func(*args):
            value = func(*args)
            gradients = []
            parents = []
            
            from itertools import count
            for i, arg, gradient in zip(count(), args, grads):
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