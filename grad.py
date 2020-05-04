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
        value = func(*args, **kwargs)

        # Reverse pass
        gradient = trace(value, var)

        return float(value), gradient
    return dfunc

def operation_overload(method):
    def new_method(self, x):
        parents = [self]
        
        value = getattr(self.super, method.__name__)(x)

        if isinstance(x, Variable):
            parents.append(x)
        else:
            parents.append(float(x))

        return Variable(value, parents=parents, operation=method.__name__)

    return new_method

class Variable(float):
    def __new__(self, *args, **kwargs):
        return super().__new__(self, *args)

    def __init__(self, *args, **kwargs):
        self.parents = kwargs.pop('parents', None)
        self.operation = kwargs.pop('operation', None)
        self.super = super()
        self.super.__init__()

    @operation_overload
    def __add__(self, x):
        pass
    @operation_overload
    def __sub__(self, x):
        pass
    @operation_overload
    def __mul__(self, x):
        pass
    @operation_overload
    def __floordiv__(self, x):
        pass
    @operation_overload
    def __truediv__(self, x):
        pass
    @operation_overload
    def __mod__(self, x):
        pass
    @operation_overload
    def __divmod__(self, x):
        pass
    @operation_overload
    def __pow__(self, x):
        pass
    @operation_overload
    def __radd__(self, x):
        pass
    @operation_overload
    def __rsub__(self, x):
        pass
    @operation_overload
    def __rmul__(self, x):
        pass
    @operation_overload
    def __rfloordiv__(self, x):
        pass
    @operation_overload
    def __rtruediv__(self, x):
        pass
    @operation_overload
    def __rmod__(self, x):
        pass
    @operation_overload
    def __rdivmod__(self, x):
        pass
    @operation_overload
    def __rpow__(self, x):
        pass
    
    def __dadd__(self):
        gradients = [float(1.), float(1.)]
        return zip(self.parents, gradients)

    def __dmul__(self):
        gradients = [float(self.parents[1]), float(self.parents[0])]
        return zip(self.parents, gradients)

    def __dradd__(self):
        gradients = [float(1.), float(1.)]
        return zip(self.parents, gradients)

    def __drmul__(self):
        gradients = [float(self.parents[1]), float(self.parents[0])]
        return zip(self.parents, gradients)

    def __str__(self):
        return '_' + super().__str__()
    pass

differential = lambda operation: operation[:2] + 'd' + operation[2:]

def trace(variable, source, gradient=1.):
    accumulation = 0.
    if variable is source:
        return gradient
    elif variable.operation is None:
        return 0.
    else:
        operation = differential(variable.operation)

    iterable = getattr(variable, operation)()

    for parent, grd in iterable:
        if isinstance(parent, Variable):
            accumulation += gradient*trace(parent, source, grd)
    
    return accumulation