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
        gradient = trace.reverse(value)

        return value.get_value(), gradient
    return dfunc

def method_overload(method):
    def new_method(self, *args, **kwargs):
        x = getattr(self.super, method.__name__)(*args, **kwargs)

        operation = method(self, x)
        if operation:
            return operation
        return x

    return new_method

def operation_overload(method):
    def new_method(self, *args, **kwargs):
        x = getattr(self.super, method.__name__)(*args, **kwargs)

        operation = method(self, x)
        if operation:
            return operation
        return x

    return new_method

class Variable(float):
    def __new__(self, *args, **kwargs):
        return super().__new__(self, *args)

    def __init__(self, *args, **kwargs):
        self.parent = kwargs.pop('parent', None)
        self.operation = kwargs.pop('operation', None)
        self.super = super()
        self.super.__init__()

    @method_overload
    def __add__(self, x):
        return Variable(x, parent=self, operation='add')
    
    def __str__(self):
        return 'V' + super().__str__()
    pass

def trace(variable):
    while True:
        print(variable , ' : ', variable.operation)
        if variable.parent is None:
            break
        variable = variable.parent