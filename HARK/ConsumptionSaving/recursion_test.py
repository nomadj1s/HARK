from math import log
from scipy.optimize import minimize_scalar

class vT(object):
    def __init__(self,c):
        self.c = c
    
    def x(self,w):
        return w
        
    def __call__(self,w):
        return self.c*log(self.x(w))

class vt(object):
    def __init__(self,c,vN):
        self.c = c
        self.vN = vN
        
    def objFunc(self,x,w):
        return -self.c*log(x) - self.vN(w - x)
    
    def x(self,w):
        x_star = minimize_scalar(self.objFunc,args=(w,),method='bounded',
                                 bounds=(1e-10,w-1e-10)).x
        return x_star
    
    def __call__(self,w):
        return self.c*log(self.x(w)) + self.vN(w - self.x(w))

p3 = vT(2.0)
p2 = vt(2.0,p3)
p1 = vt(2.0,p2)

w1 = 3.0
x1 = p1.x(w1)
w2 = w1 - x1
x2 = p2.x(w2)
w3 = w2 - x2
x3 = w3

x = [x1,x2,x3]

print('Optimal x when w1 = 3 is ' + str(x))
