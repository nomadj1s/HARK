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



start_time = clock()
d1 = [basinhopping(maxFunc,0,
                                minimizer_kwargs={"bounds":((0,self.MaxIRA),),
                                                  "args":(m,n)})
                                                     for m,n in zip(M,N)]
end_time = clock()
print('Solving without multithreading took' + mystr(end_time-start_time) + ' seconds.')

start_time = clock()
d2 = [basinhopping(self.makedvFunc,0,
                                minimizer_kwargs={"bounds":((0,self.MaxIRA),),
                                                  "args":(m,n)})
                                                     for m,n in zip(M,N)]
end_time = clock()
print('Solving with multithreading took' + mystr(end_time-start_time) + ' seconds.')

assert d1 == d2, 'Received difference answers'