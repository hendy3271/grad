method_diff = lambda operation: operation[:2] + 'd' + operation[2:]

def operation_overload(method):
    def new_method(self, *args):     
        func = getattr(self.super, method.__name__)
        gradients = getattr(self, method_diff(method.__name__))()
        return Variable(func(*args), parents=[self] + list(args), gradients=gradients)
    return new_method

class Variable(float):
    def __new__(self, *args, **kwargs):
        return super().__new__(self, *args)

    def __init__(self, *args, **kwargs):
        d_parents, d_gradients = None, None

        if len(args) > 0:
            x = args[0]
            if isinstance(x, Variable):
                d_parents = [x]
                d_gradients = [lambda x: 1.]

        self.parents = kwargs.pop('parents', d_parents)
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
        return [lambda a, b: 1., lambda b, a: 1.]

    def __dsub__(self):
        # x is this variable
        # a, b are parent one and two
        # x = a - b
        # dx/da = 1, dx/db = -1
        return [lambda a, b: 1., lambda a, b: -1.]

    def __dmul__(self):
        # x is this variable
        # a, b are parent one and two
        # x = a * b
        # dx/da = b, dx/db = a
        return [lambda a, b: b , lambda a, b: a]

    def __dtruediv__(self):
        # x is this variable
        # a, b are parent one and two
        # x = a / b
        # dx/da = 1/b, dx/db = -a/b/b
        return [lambda a, b: 1/b, lambda a, b: -a/b/b]

    def __dpow__(self):
        # x is this variable
        # a, b are parent one and two
        # x = a ^ b
        # dx/da = b*a^(b-1), dx/db = a^b*ln(a)
        from math import log
        return [lambda a, b: b*a**(b-1), lambda a, b: a**(b)*log(b)]

    def __dradd__(self):
        # refer to __dadd__
        return self.__dadd__()

    def __drsub__(self):
        # x is this variable
        # a, b are parent one and two
        # x = b - a
        # dx/da = -1, dx/db = 1
        return [lambda a, b: -1., lambda a, b: 1.]

    def __drmul__(self):
        # refer to __dmul__
        return self.__dmul__()

    def __drtruediv__(self):
        # x is this variable
        # a, b are parent one and two
        # x = b / a
        # dx/da = -b/a/a, dx/db = 1/a
        return [lambda a, b: -b/a/a, lambda a, b: 1/a]
    
    def __drpow__(self):
        # x is this variable
        # a, b are parent one and two
        # x = b ^ a
        # dx/da = b^a*ln(b), dx/db = a*b^(a-1)
        from math import log
        return [lambda a, b: b**(a)*log(b), lambda a, b: a*b**(a-1)]

    @operation_overload
    def __neg__(self): pass
    @operation_overload
    def __pos__(self): pass
    @operation_overload
    def __abs__(self): pass

    def __dneg__(self):
        # x is this variable
        # a is parent
        # x = -a
        # dx/da = -1
        return [lambda a: -1.]
    
    def __dpos__(self):
        # x is this variable
        # a is parent
        # x = +a
        # dx/da = 1
        return [lambda a: 1.]

    def __dabs__(self):
        # x is this variable
        # a is parent
        # x = a
        # dx/da = 1 if a > 0 else -1
        return [lambda a: 1. if a > 0 else -1.]

    def __str__(self):
        return '_' + super().__str__()
    pass