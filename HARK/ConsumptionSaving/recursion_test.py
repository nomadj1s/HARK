from math import log
from scipy.optimize import minimize_scalar
from functools import partial
from functools import wraps
from collections import Hashable

def memoize(obj):
    cache = obj.cache = {}

    @wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer

class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, Hashable):
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__

    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return partial(self.__call__, obj)

class vT(object):
    def __init__(self,c):
        self.c = c
    
    @memoize
    def x(self,w):
        return w
    
    @memoize    
    def __call__(self,w):
        return self.c*log(self.x(w))


class vt(object):
    def __init__(self,c,vN):
        self.c = c
        self.vN = vN
    
    @memoize    
    def objFunc(self,x,w):
        return -self.c*log(x) - self.vN(w - x)
    
    @memoize
    def x(self,w):
        x_star = minimize_scalar(self.objFunc,args=(w,),method='bounded',
                                 bounds=(1e-10,w-1e-10)).x
        return x_star
    
    @memoize
    def __call__(self,w):
        return self.c*log(self.x(w)) + self.vN(w - self.x(w))

p3 = vT(2.0)
p2 = vt(2.0,p3)
p1 = vt(2.0,p2)

#w1 = 3.0
#x1 = p1.x(w1)
#w2 = w1 - x1
#x2 = p2.x(w2)
#w3 = w2 - x2
#x3 = w3
#
#x = [x1,x2,x3]
#
#print('Optimal x when w1 = 3 is ' + str(x))
