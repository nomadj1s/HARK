'''
Classes to solve consumption-savings models with two types of savings accounts:
a liquid savings account and an IRA-like illiquid account. The illiquid account
may become liquid after a certain age, i.e.the early-withdrawal penaly
expires. All models here assume CRRA utility with geometric discounting, no
bequest motive, and income shocks are fully transitory or fully permanent. The
model incorporates a different interest rate for saving and borrowing in the
liquid account, and a separate interest rate for saving in the illiquid
account.
'''

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import range
from builtins import object
from copy import copy, deepcopy
import numpy as np
from scipy.optimize import minimzie_scalar

import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

from core import AgentType, NullFunc, HARKobject
from interpolation import CubicInterp, LowerEnvelope, LinearInterp
from simulation import drawDiscrete, drawBernoulli, drawLognormal, drawUniform
from utilities import approxMeanOneLognormal, addDiscreteOutcomeConstantMean,\
                           combineIndepDstns, makeGridExpMult, CRRAutility, CRRAutilityP, \
                           CRRAutilityPP, CRRAutilityP_inv, CRRAutility_invP, CRRAutility_inv, \
                           CRRAutilityP_invP 

utility       = CRRAutility
utilityP      = CRRAutilityP
utilityPP     = CRRAutilityPP
utilityP_inv  = CRRAutilityP_inv
utility_invP  = CRRAutility_invP
utility_inv   = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP

class ConsumerIRASolution(ConsumerSolution):
    '''
    A class representing the solution of a single period of a 
    consumption-saving model with a liquid account and an IRA-like illiquid
    savings account. The solution must include a consumption function, an
    optimal illiquid deposit function, value function and marginal value 
    function.

    Here and elsewhere in the code, Nrm indicates that variables are normalized
    by permanent income.
    '''
    distance_criteria = ['cFunc','dFunc']
    
    def __init__(self, cFunc=None, dFunc=None, vFunc=None,
                       vPfunc=None, mNrmMin=None, nNrmMin=None, 
                       hNrm=None, MPCmin=None, MPCmax=None):
        '''
        The constructor for a new ConsumerIRASolution object.

        Parameters
        ----------
        cFunc : function
            The consumption function for this period, defined over liquiud 
            market resources and illiquid account balance: c = cFunc(m,n).
        vFunc : function
            The beginning-of-period value function for this period, defined 
            over liquiud market resources and illiquid account balance: 
            v = vFunc(m,n)
        vPfunc : function
            The beginning-of-period marginal value function, with respect to
            m, for this period, defined over liquiud market resources and 
            illiquid account balance: vP = vPfunc(m,n)
        mNrmMin : float
            The minimum allowable liquid market resources for this period; 
            the consumption function (etc) are undefined for m < mNrmMin.
        nNrmMin : float
            The minimum allowable illiquid account balance for this period; 
            the consumption function (etc) are undefined for n < nNrmMin.
        hNrm : float
            Human wealth after receiving income this period: PDV of all future
            income, ignoring mortality.
        MPCmin : float
            Infimum of the marginal propensity to consume from m this period.
            MPC --> MPCmin as m --> infinity.
        MPCmax : float
            Supremum of the marginal propensity to consume from m this period.
            MPC --> MPCmax as m --> mNrmMin.

        Returns
        -------
        None
        '''
        # Change any missing function inputs to NullFunc
        if cFunc is None:
            cFunc = NullFunc()
        if dFunc is None:
            dFunc = NullFunc()
        if vFunc is None:
            vFunc = NullFunc()
        if vPfunc is None:
            vPfunc = NullFunc()
        self.cFunc        = cFunc
        self.dFunc        = dFunc
        self.vFunc        = vFunc
        self.vPfunc       = vPfunc
        self.mNrmMin      = mNrmMin
        self.nNrmMin      = nNrmMin
        self.hNrm         = hNrm
        self.MPCmin       = MPCmin
        self.MPCmax       = MPCmax