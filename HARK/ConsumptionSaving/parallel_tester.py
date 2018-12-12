'''
Testing out parallel with joblib
'''

from time import time                         # Timing utility
mystr = lambda number : "{:.4f}".format(number)# Format numbers as strings
import numpy as np 
import multiprocessing as mp
from pathos.multiprocessing import ProcessPool
from scipy.optimize import basinhopping

import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

class ParTest:
    
    def __init__(self,m,n):
        self.m = m
        self.n = n
    
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
    
    def regMax(self):
        '''
        Do maximization without parallel processing.
        '''
        m = self.m
        n = self.n
        
        d1 = [[self.findMax(mi,ni) for mi in m] for ni in n]
        self.d1 = d1
    
    def parMax(self):
        '''
        Try to do parallel processing from within a method of a class
        '''
        m = self.m
        n = self.n
        
        n_cpus = mp.cpu_count()
        pool = ProcessPool(processes=n_cpus)
        nn = np.repeat(np.array(n),len(m))
        mm = np.tile(np.array(m),len(n))
        d2 = pool.map(self.findMax, mm, nn)
        self.d2 = d2

M = np.arange(1,8,1)
N = np.arange(1,10,1)

parT = ParTest(M,N)

start_time = time()
parT.regMax()
end_time = time()
print('Solving without multithreading took ' + mystr(end_time-start_time) + ' seconds.')

start_time = time()
parT.parMax()
end_time = time()
print('Solving with multithreading took ' + mystr(end_time-start_time) + ' seconds.')

