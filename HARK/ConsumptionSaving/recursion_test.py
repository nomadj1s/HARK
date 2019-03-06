from math import log
from scipy.optimize import minimize_scalar as ms

class v3(object):
    def __init__(self,c):
        self.c = c
        
    def __call__(self,w):
        return self.c*log(w)

class v2(object):
    def __init__(self,c,v3):
        self.c = c
        self.v3 = v3
        
    def objFunc(self,x,w):
        return -self.c*log(x) - self.v3(w - x)
    
    def __call__(self,w):
        x_star = ms(self.objFunc,args=(w,),method='bounded',
                    bounds=(1e-10,w-1e-10)).x
        return self.c*log(x_star) + self.v3(w - x_star)

class v1(object):
    def __init__(self,c,v2):
        self.c = c
        self.v2 = v2
        
    def objFunc(self,x,w):
        obj =  -self.c*log(x) - self.v2(w - x)
        return obj
        
    def __call__(self,w):
        x_star = ms(self.objFunc,args=(w,),method='bounded',
                    bounds=(1e-10,w-1e-10)).x
        return x_star

