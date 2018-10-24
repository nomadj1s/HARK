'''
Testing out parallel with joblib
'''
from __future__ import print_function, division
from __future__ import absolute_import

from builtins import str
from builtins import zip
from builtins import range
from builtins import object


from time import clock                         # Timing utility
from copy import deepcopy                      # "Deep" copying for complex objects
mystr = lambda number : "{:.4f}".format(number)# Format numbers as strings
import numpy as np 
import multiprocessing as mp
from joblib import Parallel, delayed
import dill as pickle
from scipy.optimize import basinhopping

import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

from interpolation import BilinearInterp

def maxFunc(d,m,n):
    '''
    Simple function to maximize over d, given m, n.
    '''
    return (d - max(m,n))**2

def findMax(m,n):
    '''
    wrapper that uses basinhopper to maximize maxFunc
    '''
    return basinhopping(maxFunc,0,
                                minimizer_kwargs={"bounds":((-n,8),),
                                                  "args":(m,n)}).x
n_cpus = mp.cpu_count()

M = np.arange(1,8,1)
N = np.arange(1,10,1)

start_time = clock()
d1 = [[findMax(m,n) for m in M] for n in N]
end_time = clock()
print('Solving without multithreading took ' + mystr(end_time-start_time) + ' seconds.')

start_time = clock()
d2 = Parallel(n_jobs=n_cpus)(delayed(findMax)(m,n) for m in M for n in N)
end_time = clock()
print('Solving with multithreading took ' + mystr(end_time-start_time) + ' seconds.')

pool = mp.Pool(processes=n_cpus)

start_time = clock()
d3 = [pool.apply(findMax, args=(m,n)) for m in M for n in N]
end_time = clock()
print('Solving with multithreading took ' + mystr(end_time-start_time) + ' seconds.')

def unwrap_self(arg, **kwarg):
    return ParTest.findMax(*arg, **kwarg)

class ParTest:
    def maxFunc(self,d,m,n):
        '''
        Simple function to maximize over d, given m, n.
        '''
        return (d - max(m,n))**2
    
    def findMax(self,m,n):
        '''
        wrapper that uses basinhopper to maximize maxFunc
        '''
        return basinhopping(maxFunc,0,
                                minimizer_kwargs={"bounds":((-n,8),),
                                                  "args":(m,n)}).x
    
    def parMax(self,m,n):
        '''
        Try to do parallel processing from within a method of a class
        '''
        n_cpus = mp.cpu_count()
        pool = mp.Pool(processes=n_cpus)
        mm = np.repeat(np.array(m),len(n))
        nn = np.tile(np.array(n),len(m))
        d3 = [pool.apply(unwrap_self, args=(i,)) for i in zip([self]*len(mm),mm,nn)]
        self.d3 = d3

parT = ParTest()

parT.parMax(M,N)

parT.d3    
        