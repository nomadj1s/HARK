'''
Testing out parallel with joblib
'''
from __future__ import print_function, division
from __future__ import absolute_import

from builtins import zip

import numpy as np 
import multiprocessing as mp
from pathos.pools import ProcessPool
from scipy.optimize import basinhopping

import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

from scipy.interpolate import interp1d

def unwrap_self(arg, **kwarg):
    '''
    Auxiliary function needed in order to run the multiprocessing command Pool
    within a method of a class below. This gets around Pool having to call a
    method, i.e. self.findArgMaxv. Multiprocessing needs functions that can be 
    called in a global context, in order to "pickle."
    '''
    return ParTest.findMax(*arg, **kwarg)


class ParTest:
    
    def __init__(self,m,n,solution):
        self.m = m
        self.n = n
        self.solution = solution
    
    def maxFunc(self,d,m,n):
        '''
        Simple function to maximize over d, given m, n.
        '''
        return (d - max(m,n))**2
    
    def findMax(self,m,n):
        '''
        wrapper that uses basinhopper to maximize maxFunc
        '''
        return basinhopping(self.maxFunc,0.0,
                                minimizer_kwargs={"bounds":((-n,8),),
                                                  "args":(m,n)}).x
    
    def parMax(self):
        '''
        Try to do parallel processing from within a method of a class
        '''
        m = self.m
        n = self.n
        
        n_cpus = mp.cpu_count()
        pool = ProcessPool(processes=n_cpus)
        mm = np.repeat(np.array(m),len(n))
        nn = np.tile(np.array(n),len(m))
        #d3_ = [pool.apply(unwrap_self, args=(i,)) for i in zip([self]*len(mm),mm,nn)]
        d3_ = pool.map(self.findMax, mm, nn)
        d3 = np.array(d3_).reshape(len(m),len(n))
        self.d3 = d3
        
M = np.arange(1,8,1)
N = np.arange(1,10,1)

solution = interp1d(M,N[(len(N)-len(M)):])

parT = ParTest(M,N,solution)

parT.parMax()

parT.d3    
        