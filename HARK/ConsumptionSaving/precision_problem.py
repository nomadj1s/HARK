import numpy as np
from scipy.optimize import brentq as br
import itertools as itr

class maxFind(object):
    '''
    Finds argmax of f(x) = -(x - c)^2 over the interval [a,b].
    Can take numpy arrays as an argument for a and b and return an array x.
    Arrays a and b must have the same shape.
    '''
    def __init__(self,c):
        self.c = c
        
    def FOC(self,x):
        
        foc = -2*(x - self.c)
        
        return foc
    
    def __call__(self,a,b):
        
        a = np.atleast_1d(a).astype(np.float)
        b = np.atleast_1d(b).astype(np.float)
        
        x = np.full_like(a,0.0)
        
        # Corner solution at a
        corner_a = self.FOC(a) <= 0.0
        x[corner_a] = a[corner_a]
        
        # Interior solution
        interior = self.FOC(a) > 0.0 and self.FOC(b) < 0.0
        x[interior] = np.array([br(self.FOC,ai,bi) 
                                for ai,bi in itr.izip(a[interior].flatten(),
                                                      b[interior].flatten()) 
                                ]).reshape(a[interior].shape)
        
        # Corner solution at b
        corner_b = self.FOC(b) >= 0
        x[corner_b] = b[corner_b]
        
        return x


        
        
        