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

def method_overload(method):
    def new_method(self):
        parents = [self]
        
        value = getattr(self.super, method.__name__)()

        return Variable(value, parents=parents, operation=method.__name__)

    return new_method


class Variable(float):
    def __new__(self, *args, **kwargs):
        return super().__new__(self, *args)

    def __init__(self, *args, **kwargs):
        d_parents, d_operation, d_gradients = None, None, None

        if len(args) > 0:
            if isinstance(args[0], Variable):
                d_parents = [args[0]]
                d_operation = 'func'
                d_gradients = [[], [lambda *args: 1.]]

        self.parents = kwargs.pop('parents', d_parents)
        self.operation = kwargs.pop('operation', d_operation)
        self.gradients = kwargs.pop('gradients', d_gradients)
        self.super = super()
        self.super.__init__()

    @operation_overload
    def __add__(self, x): pass
    @operation_overload
    def __sub__(self, x): pass
    @operation_overload
    def __mul__(self, x): pass
    # @operation_overload
    # def __floordiv__(self, x): pass
    @operation_overload
    def __truediv__(self, x): pass
    # @operation_overload
    # def __mod__(self, x): pass
    # @operation_overload
    # def __divmod__(self, x): pass
    @operation_overload
    def __pow__(self, x): pass
    @operation_overload
    def __radd__(self, x): pass
    @operation_overload
    def __rsub__(self, x): pass
    @operation_overload
    def __rmul__(self, x): pass
    # @operation_overload
    # def __rfloordiv__(self, x): pass
    @operation_overload
    def __rtruediv__(self, x): pass
    # @operation_overload
    # def __rmod__(self, x): pass
    # @operation_overload
    # def __rdivmod__(self, x): pass
    @operation_overload
    def __rpow__(self, x): pass
    
    def __dadd__(self):
        # x is this variable
        # a, b are parent one and two
        # x = a + b
        # dx/da = 1, dx/db = 1
        gradients = [1., 1.]
        return zip(self.parents, gradients)

    def __dsub__(self):
        # x is this variable
        # a, b are parent one and two
        # x = a - b
        # dx/da = 1, dx/db = -1
        gradients = [1., -1.]
        return zip(self.parents, gradients)

    def __dmul__(self):
        # x is this variable
        # a, b are parent one and two
        # x = a * b
        # dx/da = b, dx/db = a
        gradients = [self.parents[1], self.parents[0]]
        return zip(self.parents, gradients)

    def __dtruediv__(self):
        # x is this variable
        # a, b are parent one and two
        # x = a / b
        # dx/da = 1/b, dx/db = -a/b/b
        a, b = self.parents[0], self.parents[1]
        gradients = [1/b, -a/b/b]
        return zip(self.parents, gradients)

    def __dpow__(self):
        # x is this variable
        # a, b are parent one and two
        # x = a ^ b
        # dx/da = b*a^(b-1), dx/db = a^b*ln(a)
        a, b = self.parents[0], self.parents[1]
        from math import log
        log_b = 0. if abs(b) < 1e-10 else log(b)
        gradients = [b*a**(b-1), a**(b)*log_b]
        return zip(self.parents, gradients)

    def __dradd__(self):
        return self.__dadd__()

    def __drsub__(self):
        # x is this variable
        # a, b are parent one and two
        # x = b - a
        # dx/da = 1, dx/db = -1
        gradients = [-1., 1.]
        return zip(self.parents, gradients)

    def __drmul__(self):
        return self.__dmul__()

    def __drtruediv__(self):
        # x is this variable
        # a, b are parent one and two
        # x = b / a
        # dx/da = -b/a/a, dx/db = 1/a
        a, b = self.parents[0], self.parents[1]
        gradients = [-b/a/a, 1/a]
        return zip(self.parents, gradients)
    
    def __drpow__(self):
        # x is this variable
        # a, b are parent one and two
        # x = b ^ a
        # dx/da = b^a*ln(b), dx/db = a*b^(a-1)
        a, b = self.parents[0], self.parents[1]
        from math import log
        log_b = 0. if abs(b) < 1e-10 else log(b)
        gradients = [b**(a)*log_b, a*b**(a-1)]
        return zip(self.parents, gradients)

    @method_overload
    def __neg__(self): pass
    @method_overload
    def __pos__(self): pass
    @method_overload
    def __abs__(self): pass

    def __dneg__(self):
        # x is this variable
        # a is parent
        # x = -a
        # dx/da = -1
        a = self.parents[0]
        from math import log
        gradients = [-1.]
        return zip(self.parents, gradients)
    
    def __dpos__(self):
        # x is this variable
        # a is parent
        # x = +a
        # dx/da = 1
        a = self.parents[0]
        from math import log
        gradients = [1.]
        return zip(self.parents, gradients)

    def __dabs__(self):
        # x is this variable
        # a is parent
        # x = a
        # dx/da = 1 if a > 0 else -1
        a = self.parents[0]
        from math import log
        gradients = [1. if a > 0 else -1.]
        return zip(self.parents, gradients)

    def __str__(self):
        return '_' + super().__str__()
    pass