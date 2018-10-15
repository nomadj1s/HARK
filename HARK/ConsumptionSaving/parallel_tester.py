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
import multiprocessing
from joblib import Parallel, delayed
import dill as pickle
from scipy.optimize import basinhopping

def maxFunc(d,m,n):
    '''
    Simple function to maximize over d, given m, n.
    '''
    return m*d**4 - n*d**2 + 5

def findMax(m,n):
    '''
    wrapper that uses basinhopper to maximize maxFunc
    '''
    return basinhopping(maxFunc,0,
                                minimizer_kwargs={"bounds":((-n,5),),
                                                  "args":(m,n)}).x
n_cpus = multiprocessing.cpu_count()

M = np.arange(1,10,1)
N = np.arange(1,10,1)

start_time = clock()
d1 = [[findMax(m,n) for m in M] for n in N]
end_time = clock()
print('Solving without multithreading took ' + mystr(end_time-start_time) + ' seconds.')

start_time = clock()
d2 = Parallel(n_jobs=n_cpus)(delayed(findMax)(m,n) for m in M for n in N)
end_time = clock()
print('Solving with multithreading took ' + mystr(end_time-start_time) + ' seconds.')

#assert d1 == d2, 'Received difference answers'