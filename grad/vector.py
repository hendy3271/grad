def linspace(start, stop, N = 50):
    step = (stop-start)/(N-1)
    for n in range(N):
        yield start + n*step

def vectorize(func):
    def vfunc(*args):
        args=list(args)
        maximum=0.
        for i, arg in enumerate(args):
            if isinstance(arg, (list, tuple)):
                maximum = max(maximum, len(arg))
                args[i] = arg
            else:
                from itertools import cycle
                args[i] = cycle([arg])

        vector = []

        for vargs in zip(*args):
            vector.append(func(*vargs))
            if maximum == 0.:
                break
        else:
            return vector
        return vector[0]
        
    vfunc.__vectorized__ = True
    return vfunc