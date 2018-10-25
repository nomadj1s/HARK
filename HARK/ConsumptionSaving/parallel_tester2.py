'''
Classes to solve consumption-savings models with two types of savings accounts:
a liquid savings account and an IRA-like illiquid account. The illiquid account
may become liquid after a certain age, i.e. the early-withdrawal penalty
expires. All models here assume CRRA utility with geometric discounting, no
bequest motive, and income shocks are fully transitory or fully permanent. The
model incorporates a different interest rate for saving and borrowing in the
liquid account, and a separate interest rate for saving in the illiquid
account, with a cap on deposits into the illiquid account. Consumption function
is solved for using the Nested Endogenous Grid Method (Druedahl, 2018). When
grid for illiquid asset is set to a degenerate value of zero, returns standard
consumption function.
'''

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import range
from builtins import object
from copy import copy, deepcopy
import numpy as np
from scipy.optimize import basinhopping
from time import clock, time
from joblib import Parallel, delayed
import dill as pickle
import multiprocessing as mp

import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

from core import AgentType, NullFunc, HARKobject
from interpolation import CubicInterp, LowerEnvelope, LinearInterp,\
                           BilinearInterp, ConstantFunction
from ConsIndShockModel import ConsIndShockSolver, constructAssetsGrid,\
                              IndShockConsumerType
from simulation import drawDiscrete, drawBernoulli, drawLognormal, drawUniform
from utilities import approxMeanOneLognormal, addDiscreteOutcomeConstantMean,\
                           combineIndepDstns, makeGridExpMult, CRRAutility, \
                           CRRAutilityP, CRRAutilityPP, CRRAutilityP_inv, \
                           CRRAutility_invP, CRRAutility_inv, \
                           CRRAutilityP_invP, plotFuncs 

utility       = CRRAutility
utilityP      = CRRAutilityP
utilityPP     = CRRAutilityPP
utilityP_inv  = CRRAutilityP_inv
utility_invP  = CRRAutility_invP
utility_inv   = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP

mystr = lambda number : "{:.4f}".format(number)# Format numbers as strings

M = np.arange(1,8,1)
N = np.arange(1,10,1)

def unwrap_self(arg, **kwarg):
    '''
    Auxiliary function needed in order to run the multiprocessing command Pool
    within a method of a class below. This gets around Pool having to call a
    method, i.e. self.findArgMaxv. Multiprocessing needs functions that can be 
    called in a global context, in order to "pickle."
    '''
    return ParTest.findArgMaxv(*arg, **kwarg)


class ParTest:
    
    def __init__(self,m,n,solution_next):
        self.m = m
        self.n = n
        self.MaxIRA = 8
        self.DiscFac = .98
        self.LivPrb = .98
        self.PermGroFac = 1.1
        self.CRRA = 2
        self.Rsave = 1.02
        self.Rira = 1.1
        #self.solution_next = solution_next
        
    def makeEndOfPrdvFunc(self,a,b):
        '''
        Construct the end-of-period value function for this period, storing it
        as an attribute of self for use by other methods.

        Parameters
        ----------
        EndOfPrdv : np.array
            Array of end-of-period value of assets corresponding to the
            asset values in self.aNrmNow and self.bNrmNow.

        Returns
        -------
        none
        '''
        return self.DiscFac*self.LivPrb*\
                             self.PermGroFac**(1.0-self.CRRA)*\
                             self.solution_next.vFunc(self.Rsave*a,
                                                      self.Rira*b)
        
    def makeNegvOfdFunc(self,dNrm,mNrm,nNrm):
        '''
        Constructs a beginning-period value function, given the IRA deposit (d)
        , beginning-of-period liquid resources and beginning-of-period illiquid
        assets. Since a minimizer is used, returns negative of the value
        function.
        
        Parameters
        ----------
        dNrm : float or np.array
            (Normalized) IRA deposit/withdrawal this period.
        mNrm : float or np.array
            (Normalized) liquid assets at the beginning of this period.
        nNrm : float or np.array
            (Normalized) illiquid assets at the beginning of this period.
        
        Returns
        -------
        v : float or np.array
            Negative 1 times the value function given d, m, and n.
        '''
        bNrm = nNrm + dNrm
        assert np.array(bNrm >= 0).all(), 'b should be non-negative, values' + str(dNrm) + ' ' + str(mNrm) + ' ' + str(nNrm) + ' .'
        
        lNrm = mNrm - (1 - self.PenIRA*(dNrm < 0))*dNrm
        cNrm = min(.5*lNrm,0.0001)
        aNrm = lNrm - cNrm
        
        #v = self.u(cNrm) + self.makeEndOfPrdvFunc(aNrm,bNrm)
        v = -(dNrm - max(mNrm,nNrm))**2
        
        return -v
    
    def maxFunc(self,d,m,n):
        '''
        Simple function to maximize over d, given m, n.
        '''
        return (d - max(m,n))**2
    
    def findArgMaxv(self,mNrm,nNrm):
        '''
        Wrapper function that returns d that maximizes value function given
        mNrm and nNrm.
        
        Parameters
        ----------
        mNrm : float
            (Normalized) liquid assets at the beginning of this period.
        nNrm : float
            (Normalized) illiquid assets at the beginning of this period.
            
        Returns
        -------
        d : float
            Value of d that maximizes v(d,m,n) given m and n.
        '''
        d = basinhopping(self.maxFunc,
                         0.0,minimizer_kwargs={"bounds":((-nNrm + 1e-10,
                                                          self.MaxIRA),),
                                               "args":(mNrm,nNrm)}).x
        
        return d
    
    def findMax(self,m,n):
        '''
        wrapper that uses basinhopper to maximize maxFunc
        '''
        return basinhopping(self.makeNegvOfdFunc,0.0,
                                minimizer_kwargs={"bounds":((-n,8),),
                                                  "args":(m,n)}).x
    
    def makePolicyFunc(self):
        '''
        Makes the optimal IRA deposit/withdrawal function for this period and
        optimal consumption function for this period.

        Parameters
        ----------
        cFuncNowPure : LinearInterp or BilinearInterp
            The pure consumption function for this period.
        

        Returns
        -------
        none
        '''
        mNrm = self.m
        nNrm = self.n
        
        # Use parallel processing to speed this step up
        n_cpus = mp.cpu_count()
        pool = mp.Pool(processes=n_cpus)
            
        n_repeat = np.repeat(np.array(nNrm),len(mNrm))
        m_tile = np.tile(np.array(mNrm),len(nNrm))
            
        dNrm_list = [pool.apply(unwrap_self, args=(i,)) 
                         for i in zip([self]*len(n_repeat),n_repeat,m_tile)]
            
        dNrm = np.asarray(dNrm_list).reshape(len(nNrm),len(mNrm))
        dNrm_trans = np.transpose(dNrm)
        
        self.dNrmNow = dNrm_trans

def main():
    class solution_next(HARKobject):
        def vFunc(x,y):
            return x + y
    
    parT = ParTest(M,N,solution_next)

    parT.makePolicyFunc()

    parT.dNrmNow
        
if __name__ == '__main__':
    main()