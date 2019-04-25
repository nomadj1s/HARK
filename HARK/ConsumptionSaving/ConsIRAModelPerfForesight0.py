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
from copy import copy, deepcopy
import numpy as np
from scipy.optimize import brentq as br
from functools import wraps
import pickle

import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

import matplotlib.pyplot as plt

from core import HARKobject, progress_timer
from utilities import CRRAutility, CRRAutilityP, CRRAutilityPP, \
                      CRRAutilityP_inv, CRRAutility_invP, \
                      CRRAutility_inv, CRRAutilityP_invP

utility       = CRRAutility
utilityP      = CRRAutilityP
utilityPP     = CRRAutilityPP
utilityP_inv  = CRRAutilityP_inv
utility_invP  = CRRAutility_invP
utility_inv   = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP

def memoized(obj):
    '''
    This is a wrapper. When @memoized is placed right before a method
    of a class, this causes the result of that method to be recorded. If
    the method is called again with the exact same arguments, we don't
    have to actually compute the answer again, we can just recall our
    previous value. This saves time when do simulations.
    '''
    cache = obj.cache = {}

    @wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer

class ConsIRAPFterminal0(HARKobject):
    '''
    Closed form solution for IRA consumer with perfect foresight. Solution for 
    last period.
    '''
    distance_criteria = ['CRRA','PenIRA']
    
    def __init__(self,CRRA,PenIRA=0.0,FixedCost=0.0,output='all'):
        '''
        Constructor for last period solution.
        
        Parameters
        ----------
        CRRA : float
            Coefficient of relative risk aversion.
        PenIRA: float
            Penalty for early withdrawals (d < 0) from the illiqui account, 
            i.e. before t = T_ira.
        output : string
            Whether consumption, deposit, or value or marginal value 
            function is output.
        
        Returns
        -------
        None
        '''
        self.CRRA = CRRA
        self.PenIRA = PenIRA
        self.output = output
    
    @memoized
    def __call__(self,m,n):
        '''
        Evaluate consumption decision for final period. If n and m don't
        have the same dimension, the one with the larger dimension takes
        precedent, and the first element of the one with the smaller dimension
        is used for all values of the dominent argument.
        
        Parameters
        ----------
        m : float or np.array
            Cash on hand, including period 4 income and liquid assets.
        n : float or np.array
            Illiquid account balance.
        
        Returns
        -------
        solution['cFunc'] : float or np.array
            Consumption in current period.
        solution['dFunc'] : float or np.array
            Withdrawal in current period.
        solution['aFunc'] : float or np.array
            Liquid assets at end of current period.
        solution['bFunc'] : float or np.array
            Illiquid assets at end of current period
        solution['vFunc'] : float or np.array
            Value function in current period.
        solution['vPmFunc'] : float or np.array
            Marginal value function wrt m in current period.
        solution['vPnFunc'] : float or np.array
            Marginal value function wrt n in current period.
        '''
        m = np.atleast_1d(m).astype(np.float)
        n = np.atleast_1d(n).astype(np.float)
        
        # Ensure comformability between m, n
        if m.shape != n.shape:
            if m.size >= n.size:
                n = np.full_like(m,n.item(0))
            else:
                m = np.full_like(n,m.item(0))
        
        t = self.PenIRA
        
        c = m + (1.0-t)*n
        d = -n
        a = m*0.0
        b = m*0.0
        v = utility(c,gam=self.CRRA)
        vPm = utilityP(c,gam=self.CRRA)
        vPn = utilityP(c,gam=self.CRRA)
        max_state = m*0
        max_label = np.vectorize(lambda s: 'liq')(max_state).tolist()
        
        solution = {'cFunc': c, 'dFunc': d, 'aFunc': a, 'bFunc': b, 'vFunc': v, 
                    'vPmFunc': vPm, 'vPnFunc': vPn, 'max_state': max_state,
                    'max_label': max_label}
        
        if self.output == 'all':
            return solution
        else:
            return solution[self.output]
        
class ConsIRAPFterminal(HARKobject):
    '''
    Closed form solution for IRA consumer with perfect foresight. Solution for 
    last period.
    '''
    distance_criteria = ['CRRA','PenIRA','FixedCost']
    
    def __init__(self,CRRA,PenIRA=0.0,FixedCost=0.0,output='all'):
        '''
        Constructor for last period solution.
        
        Parameters
        ----------
        CRRA : float
            Coefficient of relative risk aversion.
        PenIRA: float
            Penalty for early withdrawals (d < 0) from the illiqui account, 
            i.e. before t = T_ira.
        FixedCost: float
            Fixed utility cost of making a withdrawal from the illiquid 
            account.
        output : string
            Whether consumption, deposit, or value or marginal value 
            function is output.
        
        Returns
        -------
        None
        '''
        self.CRRA = CRRA
        self.PenIRA = PenIRA
        self.FixedCost = FixedCost
        self.output = output
    
    @memoized
    def __call__(self,m,n):
        '''
        Evaluate consumption decision for final period. If n and m don't
        have the same dimension, the one with the larger dimension takes
        precedent, and the first element of the one with the smaller dimension
        is used for all values of the dominent argument.
        
        Parameters
        ----------
        m : float or np.array
            Cash on hand, including period 4 income and liquid assets.
        n : float or np.array
            Illiquid account balance.
        
        Returns
        -------
        solution['cFunc'] : float or np.array
            Consumption in current period.
        solution['dFunc'] : float or np.array
            Withdrawal in current period.
        solution['aFunc'] : float or np.array
            Liquid assets at end of current period.
        solution['bFunc'] : float or np.array
            Illiquid assets at end of current period
        solution['vFunc'] : float or np.array
            Value function in current period.
        solution['vPmFunc'] : float or np.array
            Marginal value function wrt m in current period.
        solution['vPnFunc'] : float or np.array
            Marginal value function wrt n in current period.
        '''
        m = np.atleast_1d(m).astype(np.float)
        n = np.atleast_1d(n).astype(np.float)
        
        # Ensure comformability between m, n
        if m.shape != n.shape:
            if m.size >= n.size:
                n = np.full_like(m,n.item(0))
            else:
                m = np.full_like(n,m.item(0))
        
        t = self.PenIRA
        k = self.FixedCost
        
        c = m + (1.0-t)*n
        d = -n
        a = m*0.0
        b = m*0.0
        v = utility(c,gam=self.CRRA) - k*(n > 0.0)
        vPm = utilityP(c,gam=self.CRRA)
        vPn = (1.0-t)*utilityP(c,gam=self.CRRA)
        max_state = m*0
        max_label = np.vectorize(lambda s: 'liq')(max_state).tolist()
        
        solution = {'cFunc': c, 'dFunc': d, 'aFunc': a, 'bFunc': b, 'vFunc': v, 
                    'vPmFunc': vPm, 'vPnFunc': vPn, 'max_state': max_state,
                    'max_label': max_label}
        
        if self.output == 'all':
            return solution
        else:
            return solution[self.output]
        
class ConsIRAPFnoPen0(HARKobject):
    '''
    Solution for the IRA consumer with perfect foresight, during periods where
    there is no early withdrawal penalty.
    '''
    distance_criteria = ['period','NextIncome','DiscFac','CRRA','Rsave',
                         'Rira','MaxIRA','ConsIRAnext']
    
    def __init__(self,NextIncome,DiscFac,CRRA,Rsave,Rira,MaxIRA,
                 ConsIRAnext,FixedCost=0.0,output='all'):
        '''
        Constructor for solution in period with no early withdrawal penalty.
        
        Parameters
        ----------
        NextIncome : float or np.array
            Income in the next period
        DiscFac : float
            Intertemporal discount factor for future utility.
        CRRA : float
            Coefficient of relative risk aversion.
        Rsave: float
            Interest factor on liquid assets between this period and the 
            succeeding period when assets are positive.
        Rira:  float
            Interest factor on illiquid assets between this period and the 
            succeeding period.
        MaxIRA: float
            Maximum allowable IRA deposit, d <= MaxIRA
        ConsIRAnext : function
            Returns optimal c,d,a and value function and marginal value
            function from the next period.
        output : string
            Whether consumption, deposit, or value function is output.
        
        Returns
        -------
        None
        '''
        
        self.NextIncome     = NextIncome
        self.DiscFac        = DiscFac
        self.CRRA           = CRRA
        self.Rsave          = Rsave
        self.Rira           = Rira
        self.MaxIRA         = MaxIRA
        self.yN             = NextIncome
        self.ConsIRAnext    = deepcopy(ConsIRAnext)
        self.output         = output
    
    @memoized
    def dFOC(self,d,m,n,yN):
        '''
        Evaluate expression for d, derived from the FOC for d at an interior 
        solution. Not a closed form solution, since it also depends on d and 
        the value function of the next period.
        
        Parameters
        ----------
        d : float or np.array
            Value of d, used to calculate value function next period.
        m : float or np.array
            Cash on hand, including current period income and liquid assets.
        n : float or np.array
            Illiquid account balance.
        yN : float or np.array
            Income next period
            
        Returns
        -------
        foc : float or np.array
            Derivative of the objective function at (d,m,n,yN).
        '''
        r = self.Rira
        beta = self.DiscFac
        vPn = self.ConsIRAnext(yN,r*(n + d))['vPnFunc']
        uP = utilityP(m - d,gam=self.CRRA)
        
        foc = uP - r*beta*vPn
        
        return foc
    
    @memoized    
    def aFOC(self,a,m,n,yN):
        '''
        Evaluate expression for a, derived from the FOC for when illiquid 
        savings are capped. Not a closed form solution, since it also depends 
        on a and the value function of the next period.
        
        Parameters
        ----------
        a : float or np.array
            Value of a, used to calculate value function next period.
        m : float or np.array
            Cash on hand, including current period income and liquid assets.
        n : float or np.array
            Illiquid account balance.
        yN : float or np.array
            Income next period
            
        Returns
        -------
        foc : float or np.array
            Derivative of the objective function at (d,m,n,yN).
        '''
        r = self.Rira
        ra = self.Rsave
        beta = self.DiscFac
        dMax = self.MaxIRA
        vPm = self.ConsIRAnext(yN + ra*a,r*(n+dMax))['vPmFunc']
        uP = utilityP(m - a - dMax,gam=self.CRRA)
        
        foc = uP - ra*beta*vPm
        
        return foc
    
    @memoized
    def __call__(self,m,n):
        '''
        Evaluate optimal consupmtion, deposit, savings, value and marginal
        value functions, given liquid and illiquid assets. If n and m don't
        have the same dimension, the one with the larger dimension takes
        precedent, and the first element of the one with the smaller dimension
        is used for all values of the dominent argument. If self.yN has a
        different dimension than the dominent argument, it is likewise reduced
        to its first element.
        Parameters
        ----------
        m : float or np.array
            Cash on hand, including current period income and liquid assets.
        n : float or np.array
            Illiquid account balance.
        
        Returns
        -------
        solution['cFunc'] : float or np.array
            Consumption in current period.
        solution['dFunc'] : float or np.array
            Withdrawal in current period.
        solution['aFunc'] : float or np.array
            Liquid assets at end of current period.
        solution['bFunc'] : float or np.array
            Illiquid assets at end of current period.
        solution['vFunc'] : float or np.array
            Value function in current period.
        solution['vPmFunc'] : float or np.array
            Marginal value function wrt m in current period.
        solution['vPnFunc'] : float or np.array
            Marginal value function wrt n in current period.
        '''
        # convert to np.arrays
        m = np.atleast_1d(m).astype(np.float)
        n = np.atleast_1d(n).astype(np.float)
        yN = np.atleast_1d(self.yN).astype(np.float)

                # Ensure comformability between m, n, and yN
        if m.shape != n.shape:
            if m.size >= n.size:
                n = np.full_like(m,n.item(0))
            else:
                m = np.full_like(n,m.item(0))
        
        if yN.shape != m.shape:
            yN = np.full_like(m,yN.item(0))

        beta = self.DiscFac
        r = self.Rira
        ra = self.Rsave
        dMax = self.MaxIRA
        u = lambda c : utility(c,gam=self.CRRA)  # utility function
        uP = lambda c : utilityP(c,gam=self.CRRA) # marginal utility function
        
        s = 4 # total possible states in this period
        
        # create placeholders with arrays of dimension s, for each element of m
        c = np.reshape(np.repeat(m,s,axis=-1),m.shape + (s,)) # consumption
        d = np.full_like(c,0.0) # deposit/withdrawal
        a = np.full_like(c,0.0) # liquid savings
        b = np.full_like(c,0.0) # illiquid savings
        v = np.full_like(c,-np.inf) # value function, initiated at -inf
        vPm = np.full_like(c,1.0) # marginal value wrt m
        vPn = np.full_like(c,1.0) # marginal value wrt n
        
        # Liquidate illiquid account, no liquid savings
        solLiq = np.full_like(m,0.0)
        uPliq = np.full_like(m,0.0)
        liq = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            solLiq[idx] = self.ConsIRAnext(yN[idx],0.0)['vPnFunc']
            uPliq[idx] = uP(m[idx] + n[idx])
            
            # lower bound on withdrawal is binding
            liq[idx] = uPliq[idx] >= r*beta*solLiq[idx]
    
        if np.sum(liq) > 0: # if no one liquidates, skip
            c[...,0][liq] = m[liq] + n[liq]
            d[...,0][liq] = -n[liq]
            a[...,0][liq] = 0.0
            b[...,0][liq] = 0.0
            v[...,0][liq] = u(c[...,0][liq]) +\
                            beta*self.ConsIRAnext(yN[liq],
                                                  np.zeros(n[liq].shape)
                                                  )['vFunc']
            vPm[...,0][liq] = uP(c[...,0][liq])
            vPn[...,0][liq] = uP(c[...,0][liq])
            
        # Interior solution, partial illiquid withdrawal or deposit,
        # no liquid saving
        solCap = np.full_like(m,0.0)
        uPcap = np.full_like(m,0.0)
        inter = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            solCap[idx] = self.ConsIRAnext(yN[idx],r*(n[idx]+dMax))['vPnFunc']
            uPcap[idx] = np.where(m[idx] > dMax,uP(m[idx] - dMax),np.inf)
            
            inter[idx] = ((uPliq[idx] < r*beta*solLiq[idx]) &
                          (uPcap[idx] > r*beta*solCap[idx]))
        
        if np.sum(inter) > 0: # if no one is at interior solution, skip        
            # loop through solutions for values of m,n,yN and create an array
            for idx in np.ndindex(m.shape):
                if inter[idx]:
                    d[...,1][idx] = br(self.dFOC,-n[idx],min(dMax,m[idx]),
                                       args=(m[idx],n[idx],yN[idx]))
            
            c[...,1][inter] = m[inter] - d[...,1][inter]
            a[...,1][inter] = 0.0
            b[...,1][inter] = n[inter] + d[...,1][inter]
            v[...,1][inter] = u(c[...,1][inter]) +\
                              beta*self.ConsIRAnext(yN[inter],
                                                    r*(n[inter]
                                                    +d[...,1][inter]))['vFunc']
            vPm[...,1][inter] = uP(c[...,1][inter])
            vPn[...,1][inter] = uP(c[...,1][inter])
        
        # Iliquid savings cap & no liquid savings
        
        # upper bound on deposits and lower bound on liquid savings binds
        # cap on illiquid savings exceeds cash on hand
        solCapA = np.full_like(m,0.0)
        cap = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            solCapA[idx] = self.ConsIRAnext(yN[idx],r*(n[idx]+dMax))['vPmFunc']
            cap[idx] = ((uPcap[idx] <= r*beta*solCap[idx]) & 
                        (uPcap[idx] >= ra*beta*solCapA[idx]) &
                        (m[idx] > dMax))
        
        if np.sum(cap) > 0: # if no one is at cap w/ no liquid savings, skip               
            c[...,2][cap] = m[cap] - dMax
            d[...,2][cap] = dMax
            a[...,2][cap] = 0.0
            b[...,2][cap] = n[cap] + dMax
            v[...,2][cap] = u(c[...,2][cap]) +\
                            beta*self.ConsIRAnext(yN[cap],r*(n[cap] + dMax)
                                                  )['vFunc']
            vPm[...,2][cap] = uP(c[...,2][cap])
            vPn[...,2][cap] = r*beta*solCap[cap]
        
        # Illiquid savings cap & liquid savings
        
        # upper bound on deposits binds and lower bound on liquid savings 
        # doesn't bind
        # cap on illiquid savings exceeds cash on hand
        cap_save = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            cap_save[idx] = ((uPcap[idx] <= r*beta*solCap[idx]) & 
                             (uPcap[idx] < ra*beta*solCapA[idx]) &
                             (m[idx] > dMax))
        
        if np.sum(cap_save) > 0: # if no one is at cap w/ liquid savings, skip
            # loop through solutions for values of m,n,yN and create an array
            for idx in np.ndindex(m.shape):
                if cap_save[idx]:
                    a[...,3][idx] = br(self.aFOC,0.0,m[idx]-dMax,
                                       args=(m[idx],n[idx],yN[idx]))
                                           
            solCapSave = self.ConsIRAnext(yN[cap_save]+ra*a[...,3][cap_save],
                                          r*(n[cap_save]+dMax))
        
            c[...,3][cap_save] = m[cap_save] - dMax - a[...,3][cap_save]
            d[...,3][cap_save] = dMax
            b[...,3][cap_save] = n[cap_save] + dMax
            v[...,3][cap_save] = u(c[...,3][cap_save]) +\
                                 beta*solCapSave['vFunc']
            vPm[...,3][cap_save] = uP(c[...,3][cap_save])
            vPn[...,3][cap_save] = r*beta*solCapSave['vPnFunc']
        
        # Find index of max utility among valid solutions
        max_state = np.argmax(v,axis=-1)
        
        # Create tuple of common dimensions for indexing max values
        max_dim = np.ogrid[[slice(i) for i in max_state.shape]]
        
        # Select elements from each array, based on index of max value
        c_star = c[tuple(max_dim) + (max_state,)]
        d_star = d[tuple(max_dim) + (max_state,)]
        a_star = a[tuple(max_dim) + (max_state,)]
        b_star = b[tuple(max_dim) + (max_state,)]
        v_star = v[tuple(max_dim) + (max_state,)]
        vPm_star = vPm[tuple(max_dim) + (max_state,)]
        vPn_star = vPn[tuple(max_dim) + (max_state,)]
        
        solution = {'cFunc': c_star, 'dFunc': d_star, 'aFunc': a_star,
                    'bFunc': b_star, 'vFunc': v_star, 'vPmFunc': vPm_star, 
                    'vPnFunc': vPn_star, 'max_state': max_state}
        
        if self.output == 'all':
            return solution
        else:
            return solution[self.output]
        
class ConsIRAPFpen0(HARKobject):
    '''
    Solution for the IRA consumer with perfect foresight, during periods where
    there is an early withdrawal penalty.
    '''
    distance_criteria = ['period','NextIncome','DiscFac','CRRA','Rsave',
                         'Rira','PenIRA','MaxIRA','ConsIRAnext']
    
    def __init__(self,NextIncome,DiscFac,CRRA,Rsave,Rira,PenIRA,MaxIRA,
                 ConsIRAnext,FixedCost=0.0,output='all'):
        '''
        Constructor for solution in period with an early withdrawal penalty.
        
        Parameters
        ----------
        NextIncome : float or np.array
            Income in the next period.
        DiscFac : float
            Intertemporal discount factor for future utility.
        CRRA : float
            Coefficient of relative risk aversion.
        Rsave: float
            Interest factor on liquid assets between this period and the 
            succeeding period when assets are positive.
        Rira:  float
            Interest factor on illiquid assets between this period and the 
            succeeding period.
        PenIRA: float
            Penalty for early withdrawals (d < 0) from the illiqui account, 
            i.e. before t = T_ira.
        MaxIRA: float
            Maximum allowable IRA deposit, d <= MaxIRA
        ConsIRAnext : function
            Returns optimal c,d,a and value function and marginal value
            function from the next period.
        output : string
            Whether consumption, deposit, or value function is output.
        
        Returns
        -------
        None
        '''
        
        self.NextIncome     = NextIncome
        self.DiscFac        = DiscFac
        self.CRRA           = CRRA
        self.Rsave          = Rsave
        self.Rira           = Rira
        self.PenIRA         = PenIRA
        self.MaxIRA         = MaxIRA
        self.yN             = NextIncome
        self.ConsIRAnext    = deepcopy(ConsIRAnext)
        self.output         = output
    
    @memoized    
    def wFOC(self,d,m,n,yN):
        '''
        Evaluate expression for d, derived from the FOC for d in period 1 at
        an interior solution when making a withdrawal. Not a closed form 
        solution, since it also depends on d and the value function of the 
        next period.
        
        Parameters
        ----------
        d : float or np.array
            Value of d, used to calculate value function next period.
        m : float or np.array
            Cash on hand, including period 1 income and liquid assets.
        n : float or np.array
            Illiquid account balance.
        yN : float or np.array
            Income next period
            
        Returns
        -------
        foc : float or np.array
            Derivative of the objective function at (d,m,n,yN).
        '''
        r = self.Rira
        t = self.PenIRA
        beta = self.DiscFac
        vPn = self.ConsIRAnext(yN,r*(n + d))['vPnFunc']
        uP = (1.0-t)*utilityP(m - (1.0-t)*d,gam=self.CRRA)
        
        foc = uP - r*beta*vPn
        
        return foc
    
    @memoized
    def dFOC(self,d,m,n,yN):
        '''
        Evaluate expression for d, derived from the FOC for d in period 1 at
        an interior solution when making a deposit. Not a closed form 
        solution, since it also depends on d and the value function of the 
        next period.
        
        Parameters
        ----------
        d : float or np.array
            Value of d, used to calculate value function next period.
        m : float or np.array
            Cash on hand, including period 1 income and liquid assets.
        n : float
            Illiquid account balance.
        yN : float or np.array
            Income next period
            
        Returns
        -------
        foc : float or np.array
            Derivative of the objective function at (d,m,n,yN).
        '''
        r = self.Rira
        beta = self.DiscFac
        vPn = self.ConsIRAnext(yN,r*(n + d))['vPnFunc']
        uP = utilityP(m - d,gam=self.CRRA)
        
        foc = uP - r*beta*vPn
        
        return foc
    
    @memoized
    def aKinkFOC(self,a,m,n,yN):
        '''
        Evaluate expression for a, derived from the FOC for a when illiquid 
        savings are at a kink. Not a closed form solution, since it 
        also depends on a and the value function of the next period.
        
        Parameters
        ----------
        a : float or np.array
            Value of a, used to calculate value function next period.
        m : float or np.array
            Cash on hand, including period 1 income and liquid assets.
        n : float or np.array
            Illiquid account balance.
        yN : float or np.array
            Income next period
            
        Returns
        -------
        foc : float or np.array
            Derivative of the objective function at (a,m,n,yN).
        '''
        r = self.Rira
        ra = self.Rsave
        beta = self.DiscFac
        vPm = self.ConsIRAnext(yN + ra*a,r*n)['vPmFunc']
        uP = utilityP(m - a,gam=self.CRRA)
        
        foc = uP - ra*beta*vPm
        
        return foc
    
    @memoized
    def aFOC(self,a,m,n,yN):
        '''
        Evaluate the FOC for a when illiquid savings are capped.
        
        Parameters
        ----------
        a : float or np.array
            Value of a, used to calculate value function next period.
        m : float or np.array
            Cash on hand, including period 1 income and liquid assets.
        n : float or np.array
            Illiquid account balance.
        yN : float or np.array
            Income next period
            
        Returns
        -------
        foc : float or np.array
            Derivative of the objective function at (a,m,n,yN).
        '''
        r = self.Rira
        ra = self.Rsave
        beta = self.DiscFac
        dMax = self.MaxIRA
        vPm = self.ConsIRAnext(yN + ra*a,r*(n+dMax))['vPmFunc']
        uP = utilityP(m - a - dMax,gam=self.CRRA)
        
        foc = uP - ra*beta*vPm
        
        return foc
    
    @memoized
    def __call__(self,m,n):
        '''
        Evaluate optimal consupmtion, deposit, savings, value and marginal
        value functions, given liquid and illiquid assets. If n and m don't
        have the same dimension, the one with the larger dimension takes
        precedent, and the first element of the one with the smaller dimension
        is used for all values of the dominent argument. If self.yN has a
        different dimension than the dominent argument, it is likewise reduced
        to its first element.
        
        Parameters
        ----------
        m : float or np.array
            Cash on hand, including current period income and liquid assets.
        n : float or np.array
            Illiquid account balance.
        
        Returns
        -------
        solution['cFunc'] : float or np.array
            Consumption in current period.
        solution['dFunc'] : float or np.array
            Withdrawal in current period.
        solution['aFunc'] : float or np.array
            Liquid assets at end of current period.
        solution['bFunc'] : float or np.array
            Illiquid assets at end of current period.
        solution['vFunc'] : float or np.array
            Value function in current period.
        solution['vPmFunc'] : float or np.array
            Marginal value function wrt m in current period.
        solution['vPnFunc'] : float or np.array
            Marginal value function wrt n in current period.
        '''
        # convert to np.arrays
        m = np.atleast_1d(m).astype(np.float)
        n = np.atleast_1d(n).astype(np.float)
        yN = np.atleast_1d(self.yN).astype(np.float)

        # Ensure comformability between m, n, and yN
        if m.shape != n.shape:
            if m.size >= n.size:
                n = np.full_like(m,n.item(0))
            else:
                m = np.full_like(n,m.item(0))
        
        if yN.shape != m.shape:
            yN = np.full_like(m,yN.item(0))
    
        beta = self.DiscFac
        r = self.Rira
        t = self.PenIRA
        ra = self.Rsave
        dMax = self.MaxIRA
        u = lambda c : utility(c,gam=self.CRRA)  # utility function
        uP = lambda c: utilityP(c,gam=self.CRRA) # marginal utility function
        
        s = 7 # total possible states in this period
        
        # create placeholders with arrays of dimension s, for each element of m
        c = np.reshape(np.repeat(m,s,axis=-1),m.shape + (s,)) # consumption
        d = np.full_like(c,0.0) # deposit/withdrawal
        a = np.full_like(c,0.0) # liquid savings
        b = np.full_like(c,0.0) # illiquid savings
        v = np.full_like(c,-np.inf) # value function, initiated at -inf
        vPm = np.full_like(c,1.0) # marginal value wrt m
        vPn = np.full_like(c,1.0) # marginal value wrt n
               
        # Liquidate illiquid account, no liquid savings
        solLiq = np.full_like(m,0.0)
        uPliq = np.full_like(m,0.0)
        liq = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            solLiq[idx] = self.ConsIRAnext(yN[idx],0.0)['vPnFunc']
            uPliq[idx] = (1.0-t)*uP(m[idx] + (1.0-t)*n[idx])

            # lower bound on withdrawal is binding        
            liq[idx] = uPliq[idx] >= r*beta*solLiq[idx]
        
        if np.sum(liq) > 0: # if no one liquidates, skip
            c[...,0][liq] = m[liq] + (1.0-t)*n[liq]
            d[...,0][liq] = -n[liq]
            a[...,0][liq] = 0.0
            b[...,0][liq] = 0.0
            v[...,0][liq] = u(c[...,0][liq]) +\
                            beta*self.ConsIRAnext(yN[liq],
                                                  np.zeros(n[liq].shape)
                                                  )['vFunc']
            vPm[...,0][liq] = uP(c[...,0][liq])
            vPn[...,0][liq] = (1.0-t)*uP(c[...,0][liq])       
            
        # Interior solution, partial illiquid withdrawal, no liquid saving
        solKink = np.full_like(m,0.0)
        uPwithdr = np.full_like(m,0.0)
        withdr = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            solKink[idx] = self.ConsIRAnext(yN[idx],r*n[idx])['vPnFunc']
            uPwithdr[idx] = (1.0-t)*uP(m[idx])
        
            # neither bound on withdrawals binds
            withdr[idx] = ((uPliq[idx] < r*beta*solLiq[idx]) & 
                           (uPwithdr[idx] > r*beta*solKink[idx]))
        
        # interior solution for withdrawal
        if np.sum(withdr) > 0: # if no one withdraws, skip
            # loop through solutions for values of m,n,yN and create an array
            for idx in np.ndindex(m.shape):
                if withdr[idx]:
                    d[...,1][idx] = br(self.wFOC,-n[idx],0.0,
                                       args=(m[idx],n[idx],yN[idx])) 
        
            c[...,1][withdr] = m[withdr] - (1.0-t)*d[...,1][withdr]
            a[...,1][withdr] = 0.0
            b[...,1][withdr] = n[withdr] + d[...,1][withdr]
            v[...,1][withdr] = u(c[...,1][withdr]) +\
                               beta*self.ConsIRAnext(yN[withdr],
                                                     r*(n[withdr]
                                                     +d[...,1][withdr])
                                                     )['vFunc']
            vPm[...,1][withdr] = uP(c[...,1][withdr])
            vPn[...,1][withdr] = (1.0-t)*uP(c[...,1][withdr])
        
        # Corner solution w/ no illiquid withdrawal or saving, no liquid saving
        uPdep = np.full_like(m,0.0)
        solKinkA = np.full_like(m,0.0)
        kink = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            uPdep[idx] = uP(m[idx])
            solKinkA[idx] = self.ConsIRAnext(yN[idx],r*n[idx])['vPmFunc']
        
            # upperbound on withdrawals and lower bound on deposits bind
            # lower bound on liquid savings binds
            kink[idx] = ((uPwithdr[idx] <= r*beta*solKink[idx]) &
                         (uPdep[idx] >= r*beta*solKink[idx]) &
                         (uPdep[idx] >= ra*beta*solKinkA[idx]))
        
        if np.sum(kink) > 0: # if no one is at kink, skip
            c[...,2][kink] = m[kink]
            d[...,2][kink] = 0.0
            a[...,2][kink] = 0.0
            b[...,2][kink] = n[kink]
            v[...,2][kink] = u(c[...,2][kink]) +\
                             beta*self.ConsIRAnext(yN[kink],r*n[kink])['vFunc']
            vPm[...,2][kink] = uP(c[...,2][kink])
            vPn[...,2][kink] = r*beta*solKink[kink]
            
        # Corner solution w/ no illiquid withdrawal or saving & liquid saving
        
        # upperbound on withdrawals and lower bound on deposits bind
        # lower bound on liquid savings doesn't bind
        kink_save = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            kink_save[idx] = ((uPwithdr[idx] <= r*beta*solKink[idx]) &
                              (uPdep[idx] >= r*beta*solKink[idx]) &
                              (uPdep[idx] < ra*beta*solKinkA[idx]))
        
        if np.sum(kink_save) > 0: # if no one saves at kink, skip
            # loop through solutions for values of m,n,yN and create an array
            for idx in np.ndindex(m.shape):
                if kink_save[idx]:
                    a[...,3][idx] = br(self.aKinkFOC,0.0,m[idx],
                                       args=(m[idx],n[idx],yN[idx]))       
        
            solKinkSave = self.ConsIRAnext(yN[kink_save] 
                                           + ra*a[...,3][kink_save],
                                           r*n[kink_save])
        
            c[...,3][kink_save] = m[kink_save] - a[...,3][kink_save]
            d[...,3][kink_save] = 0.0
            b[...,3][kink_save] = n[kink_save]
            v[...,3][kink_save] = u(c[...,3][kink_save]) +\
                                  beta*solKinkSave['vFunc']
            vPm[...,3][kink_save] = uP(c[...,3][kink_save])
            vPn[...,3][kink_save] = r*beta*solKinkSave['vPnFunc']
            
        # Interior solution, partial liquid deposit, no liquid saving
        solCap = np.full_like(m,0.0)
        uPcap = np.full_like(m,0.0)
        dep = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            solCap[idx] = self.ConsIRAnext(yN[idx],r*(n[idx]+dMax))['vPnFunc']
            # if m <= dMax, will not reach illiquid savings cap
            uPcap[idx] = np.where(m[idx] > dMax,uP(m[idx] - dMax),np.inf)
            
            # neither bound is binding for deposits
            dep[idx] = ((uPdep[idx] < r*beta*solKink[idx]) & 
                        (uPcap[idx] > r*beta*solCap[idx]))
        
        if np.sum(dep) > 0: # if no one deposits, skip
            # loop through solutions for values of m,n,yN and create an array
            for idx in np.ndindex(m.shape):
                if dep[idx]:
                    d[...,4][idx] = br(self.dFOC,0.0,min(dMax,m[idx]),
                                       args=(m[idx],n[idx],yN[idx]))
        
            c[...,4][dep] = m[dep] - d[...,4][dep]
            a[...,4][dep] = 0.0
            b[...,4][dep] = n[dep] + d[...,4][dep]
            v[...,4][dep] = u(c[...,4][dep]) +\
                            beta*self.ConsIRAnext(yN[dep],
                                                  r*(n[dep] 
                                                  +d[...,4][dep]))['vFunc']
            vPm[...,4][dep] = uP(c[...,4][dep])
            vPn[...,4][dep] = uP(c[...,4][dep])
                    
        # Illiquid savings at cap, no liquid savings
        
        # upper bound on deposits and lower bound on liquid savings binds
        # cap on illiquid savings exceeds cash on hand
        solCapA = np.full_like(m,0.0)
        cap = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            solCapA[idx] = self.ConsIRAnext(yN[idx],r*(n[idx]+dMax))['vPmFunc']
            cap[idx] = ((uPcap[idx] <= r*beta*solCap[idx]) & 
                        (uPcap[idx] >= ra*beta*solCapA[idx]) &
                        (m[idx] > dMax))
        
        if np.sum(cap) > 0: # if no one is at cap w/ no liquid savings, skip
            c[...,5][cap] = m[cap] - dMax
            d[...,5][cap] = dMax
            a[...,5][cap] = 0.0
            b[...,5][cap] = n[cap] + dMax
            v[...,5][cap] = u(c[...,5][cap]) +\
                            beta*self.ConsIRAnext(yN[cap],r*(n[cap]+dMax)
                                                  )['vFunc']
            vPm[...,5][cap] = uP(c[...,5][cap])
            vPn[...,5][cap] = r*beta*solCap[cap]
            
        # Illiquid savings cap and liquid savings
        
        # upper bound on deposits binds and lower bound on liquid savings 
        # doesn't bind
        # cap on illiquid savings exceeds cash on hand
        cap_save = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            cap_save[idx] = ((uPcap[idx] <= r*beta*solCap[idx]) & 
                             (uPcap[idx] < ra*beta*solCapA[idx]) &
                             (m[idx] > dMax))
        
        
        if np.sum(cap_save) > 0: # if no one is at cap w/ liquid savings, skip
            # loop through solutions for values of m,n,yN and create an array
            for idx in np.ndindex(m.shape):
                if cap_save[idx]:
                    a[...,6][idx] = br(self.aFOC,0.0,m[idx]-dMax,
                                       args=(m[idx],n[idx],yN[idx]))
        
            solCapSave = self.ConsIRAnext(yN[cap_save]+ra*a[...,6][cap_save],
                                          r*(n[cap_save]+dMax))
        
            c[...,6][cap_save] = m[cap_save] - dMax - a[...,6][cap_save]
            d[...,6][cap_save] = dMax
            b[...,6][cap_save] = n[cap_save] + dMax
            v[...,6][cap_save] = u(c[...,6][cap_save]) +\
                                 beta*solCapSave['vFunc']
            vPm[...,6][cap_save] = uP(c[...,6][cap_save])
            vPn[...,6][cap_save] = r*beta*solCapSave['vPnFunc']
        
        # Find index of max utility among valid solutions
        max_state = np.argmax(v,axis=-1)
        
        # Create tuple of common dimensions for indexing max values
        max_dim = np.ogrid[[slice(i) for i in max_state.shape]]
        
        # Select elements from each array, based on index of max value
        c_star = c[tuple(max_dim) + (max_state,)]
        d_star = d[tuple(max_dim) + (max_state,)]
        a_star = a[tuple(max_dim) + (max_state,)]
        b_star = b[tuple(max_dim) + (max_state,)]
        v_star = v[tuple(max_dim) + (max_state,)]
        vPm_star = vPm[tuple(max_dim) + (max_state,)]
        vPn_star = vPn[tuple(max_dim) + (max_state,)]
        
        labDict = {0:'liq',1:'withdr',2:'kink',3:'kink_save',4:'dep',
                   5:'cap',6:'cap_save'}
        
        max_label = np.vectorize(lambda s: labDict[s])(max_state).tolist()
        
        solution = {'cFunc': c_star, 'dFunc': d_star, 'aFunc': a_star,
                    'bFunc': b_star, 'vFunc': v_star, 'vPmFunc': vPm_star,
                    'vPnFunc': vPn_star, 'max_state': max_state,
                    'max_label': max_label}
        
        if self.output == 'all':
            return solution
        else:
            return solution[self.output]
        
class ConsIRAPFpen(HARKobject):
    '''
    Solution for the IRA consumer with perfect foresight, during periods where
    there is an early withdrawal penalty.
    '''
    distance_criteria = ['period','NextIncome','DiscFac','CRRA','Rsave',
                         'Rira','PenIRA','MaxIRA','FixedCost','ConsIRAnext']
    
    def __init__(self,NextIncome,DiscFac,CRRA,Rsave,Rira,PenIRA,MaxIRA,
                 FixedCost,ConsIRAnext,output='all'):
        '''
        Constructor for solution in period with an early withdrawal penalty.
        
        Parameters
        ----------
        NextIncome : float or np.array
            Income in the next period.
        DiscFac : float
            Intertemporal discount factor for future utility.
        CRRA : float
            Coefficient of relative risk aversion.
        Rsave: float
            Interest factor on liquid assets between this period and the 
            succeeding period when assets are positive.
        Rira:  float
            Interest factor on illiquid assets between this period and the 
            succeeding period.
        PenIRA: float
            Penalty for early withdrawals (d < 0) from the illiqui account, 
            i.e. before t = T_ira.
        MaxIRA: float
            Maximum allowable IRA deposit, d <= MaxIRA
                FixedCost: float
        Fixed utility cost of making a withdrawal from the illiquid 
            account. 
        ConsIRAnext : function
            Returns optimal c,d,a and value function and marginal value
            function from the next period.
        output : string
            Whether consumption, deposit, or value function is output.
        
        Returns
        -------
        None
        '''
        
        self.NextIncome     = NextIncome
        self.DiscFac        = DiscFac
        self.CRRA           = CRRA
        self.Rsave          = Rsave
        self.Rira           = Rira
        self.PenIRA         = PenIRA
        self.MaxIRA         = MaxIRA
        self.FixedCost      = FixedCost
        self.yN             = NextIncome
        self.ConsIRAnext    = deepcopy(ConsIRAnext)
        self.output         = output
    
    @memoized    
    def wFOC(self,d,m,n,yN):
        '''
        Evaluate expression for d, derived from the FOC for d in period 1 at
        an interior solution when making a withdrawal. Not a closed form 
        solution, since it also depends on d and the value function of the 
        next period.
        
        Parameters
        ----------
        d : float or np.array
            Value of d, used to calculate value function next period.
        m : float or np.array
            Cash on hand, including period 1 income and liquid assets.
        n : float or np.array
            Illiquid account balance.
        yN : float or np.array
            Income next period
            
        Returns
        -------
        foc : float or np.array
            Derivative of the objective function at (d,m,n,yN).
        '''
        r = self.Rira
        t = self.PenIRA
        beta = self.DiscFac
        vPn = self.ConsIRAnext(yN,r*(n + d))['vPnFunc']
        uP = (1.0-t)*utilityP(m - (1.0-t)*d,gam=self.CRRA)
        
        foc = uP - r*beta*vPn
        
        return foc
    
    @memoized
    def dFOC(self,d,m,n,yN):
        '''
        Evaluate expression for d, derived from the FOC for d in period 1 at
        an interior solution when making a deposit. Not a closed form 
        solution, since it also depends on d and the value function of the 
        next period.
        
        Parameters
        ----------
        d : float or np.array
            Value of d, used to calculate value function next period.
        m : float or np.array
            Cash on hand, including period 1 income and liquid assets.
        n : float
            Illiquid account balance.
        yN : float or np.array
            Income next period
            
        Returns
        -------
        foc : float or np.array
            Derivative of the objective function at (d,m,n,yN).
        '''
        r = self.Rira
        beta = self.DiscFac
        vPn = self.ConsIRAnext(yN,r*(n + d))['vPnFunc']
        uP = utilityP(m - d,gam=self.CRRA)
        
        foc = uP - r*beta*vPn
        
        return foc
    
    @memoized
    def aKinkFOC(self,a,m,n,yN):
        '''
        Evaluate expression for a, derived from the FOC for a when illiquid 
        savings are at a kink. Not a closed form solution, since it 
        also depends on a and the value function of the next period.
        
        Parameters
        ----------
        a : float or np.array
            Value of a, used to calculate value function next period.
        m : float or np.array
            Cash on hand, including period 1 income and liquid assets.
        n : float or np.array
            Illiquid account balance.
        yN : float or np.array
            Income next period
            
        Returns
        -------
        foc : float or np.array
            Derivative of the objective function at (a,m,n,yN).
        '''
        r = self.Rira
        ra = self.Rsave
        beta = self.DiscFac
        vPm = self.ConsIRAnext(yN + ra*a,r*n)['vPmFunc']
        uP = utilityP(m - a,gam=self.CRRA)
        
        foc = uP - ra*beta*vPm
        
        return foc
    
    @memoized
    def aFOC(self,a,m,n,yN):
        '''
        Evaluate the FOC for a when illiquid savings are capped.
        
        Parameters
        ----------
        a : float or np.array
            Value of a, used to calculate value function next period.
        m : float or np.array
            Cash on hand, including period 1 income and liquid assets.
        n : float or np.array
            Illiquid account balance.
        yN : float or np.array
            Income next period
            
        Returns
        -------
        foc : float or np.array
            Derivative of the objective function at (a,m,n,yN).
        '''
        r = self.Rira
        ra = self.Rsave
        beta = self.DiscFac
        dMax = self.MaxIRA
        vPm = self.ConsIRAnext(yN + ra*a,r*(n+dMax))['vPmFunc']
        uP = utilityP(m - a - dMax,gam=self.CRRA)
        
        foc = uP - ra*beta*vPm
        
        return foc
    
    @memoized
    def __call__(self,m,n):
        '''
        Evaluate optimal consupmtion, deposit, savings, value and marginal
        value functions, given liquid and illiquid assets. If n and m don't
        have the same dimension, the one with the larger dimension takes
        precedent, and the first element of the one with the smaller dimension
        is used for all values of the dominent argument. If self.yN has a
        different dimension than the dominent argument, it is likewise reduced
        to its first element.
        
        Parameters
        ----------
        m : float or np.array
            Cash on hand, including current period income and liquid assets.
        n : float or np.array
            Illiquid account balance.
        
        Returns
        -------
        solution['cFunc'] : float or np.array
            Consumption in current period.
        solution['dFunc'] : float or np.array
            Withdrawal in current period.
        solution['aFunc'] : float or np.array
            Liquid assets at end of current period.
        solution['bFunc'] : float or np.array
            Illiquid assets at end of current period.
        solution['vFunc'] : float or np.array
            Value function in current period.
        solution['vPmFunc'] : float or np.array
            Marginal value function wrt m in current period.
        solution['vPnFunc'] : float or np.array
            Marginal value function wrt n in current period.
        '''
        # convert to np.arrays
        m = np.atleast_1d(m).astype(np.float)
        n = np.atleast_1d(n).astype(np.float)
        yN = np.atleast_1d(self.yN).astype(np.float)

        # Ensure comformability between m, n, and yN
        if m.shape != n.shape:
            if m.size >= n.size:
                n = np.full_like(m,n.item(0))
            else:
                m = np.full_like(n,m.item(0))
        
        if yN.shape != m.shape:
            yN = np.full_like(m,yN.item(0))
    
        beta = self.DiscFac
        r = self.Rira
        t = self.PenIRA
        ra = self.Rsave
        dMax = self.MaxIRA
        k = self.FixedCost
        u = lambda c : utility(c,gam=self.CRRA)  # utility function
        uP = lambda c: utilityP(c,gam=self.CRRA) # marginal utility function
        
        s = 7 # total possible states in this period
        
        # create placeholders with arrays of dimension s, for each element of m
        c = np.reshape(np.repeat(m,s,axis=-1),m.shape + (s,)) # consumption
        d = np.full_like(c,0.0) # deposit/withdrawal
        a = np.full_like(c,0.0) # liquid savings
        b = np.full_like(c,0.0) # illiquid savings
        v = np.full_like(c,-np.inf) # value function, initiated at -inf
        vPm = np.full_like(c,1.0) # marginal value wrt m
        vPn = np.full_like(c,1.0) # marginal value wrt n
               
        # Liquidate illiquid account, no liquid savings
        solLiq = np.full_like(m,0.0)
        uPliq = np.full_like(m,0.0)
        liq = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            solLiq[idx] = self.ConsIRAnext(yN[idx],0.0)['vPnFunc']
            uPliq[idx] = (1.0-t)*uP(m[idx] + (1.0-t)*n[idx])

            # lower bound on withdrawal is binding        
            liq[idx] = uPliq[idx] >= r*beta*solLiq[idx]
        
        if np.sum(liq) > 0: # if no one liquidates, skip
            c[...,0][liq] = m[liq] + (1.0-t)*n[liq]
            d[...,0][liq] = -n[liq]
            a[...,0][liq] = 0.0
            b[...,0][liq] = 0.0
            v[...,0][liq] = u(c[...,0][liq]) - k +\
                            beta*self.ConsIRAnext(yN[liq],
                                                  np.zeros(n[liq].shape)
                                                  )['vFunc']
            vPm[...,0][liq] = uP(c[...,0][liq])
            vPn[...,0][liq] = (1.0-t)*uP(c[...,0][liq])       
            
        # Interior solution, partial illiquid withdrawal, no liquid saving
        solKink = np.full_like(m,0.0)
        uPwithdr = np.full_like(m,0.0)
        withdr = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            solKink[idx] = self.ConsIRAnext(yN[idx],r*n[idx])['vPnFunc']
            uPwithdr[idx] = (1.0-t)*uP(m[idx])
        
            # neither bound on withdrawals binds
            withdr[idx] = ((uPliq[idx] < r*beta*solLiq[idx]) & 
                           (uPwithdr[idx] > r*beta*solKink[idx]))
        
        # interior solution for withdrawal
        if np.sum(withdr) > 0: # if no one withdraws, skip
            # loop through solutions for values of m,n,yN and create an array
            for idx in np.ndindex(m.shape):
                if withdr[idx]:
                    d[...,1][idx] = br(self.wFOC,-n[idx],0.0,
                                       args=(m[idx],n[idx],yN[idx])) 
        
            c[...,1][withdr] = m[withdr] - (1.0-t)*d[...,1][withdr]
            a[...,1][withdr] = 0.0
            b[...,1][withdr] = n[withdr] + d[...,1][withdr]
            v[...,1][withdr] = u(c[...,1][withdr]) - k +\
                               beta*self.ConsIRAnext(yN[withdr],
                                                     r*(n[withdr]
                                                     +d[...,1][withdr])
                                                     )['vFunc']
            vPm[...,1][withdr] = uP(c[...,1][withdr])
            vPn[...,1][withdr] = (1.0-t)*uP(c[...,1][withdr])
        
        # Corner solution w/ no illiquid withdrawal or saving, no liquid saving
        uPdep = np.full_like(m,0.0)
        solKinkA = np.full_like(m,0.0)
        kink = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            uPdep[idx] = uP(m[idx])
            solKinkA[idx] = self.ConsIRAnext(yN[idx],r*n[idx])['vPmFunc']
        
            # upperbound on withdrawals and lower bound on deposits bind
            # lower bound on liquid savings binds
            kink[idx] = ((uPwithdr[idx] <= r*beta*solKink[idx]) &
                         (uPdep[idx] >= r*beta*solKink[idx]) &
                         (uPdep[idx] >= ra*beta*solKinkA[idx]))
        
        if np.sum(kink) > 0: # if no one is at kink, skip
            c[...,2][kink] = m[kink]
            d[...,2][kink] = 0.0
            a[...,2][kink] = 0.0
            b[...,2][kink] = n[kink]
            v[...,2][kink] = u(c[...,2][kink]) +\
                             beta*self.ConsIRAnext(yN[kink],r*n[kink])['vFunc']
            vPm[...,2][kink] = uP(c[...,2][kink])
            vPn[...,2][kink] = r*beta*solKink[kink]
            
        # Corner solution w/ no illiquid withdrawal or saving & liquid saving
        
        # upperbound on withdrawals and lower bound on deposits bind
        # lower bound on liquid savings doesn't bind
        kink_save = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            kink_save[idx] = ((uPwithdr[idx] <= r*beta*solKink[idx]) &
                              (uPdep[idx] >= r*beta*solKink[idx]) &
                              (uPdep[idx] < ra*beta*solKinkA[idx]))
        
        if np.sum(kink_save) > 0: # if no one saves at kink, skip
            # loop through solutions for values of m,n,yN and create an array
            for idx in np.ndindex(m.shape):
                if kink_save[idx]:
                    a[...,3][idx] = br(self.aKinkFOC,0.0,m[idx],
                                       args=(m[idx],n[idx],yN[idx]))       
        
            solKinkSave = self.ConsIRAnext(yN[kink_save] 
                                           + ra*a[...,3][kink_save],
                                           r*n[kink_save])
        
            c[...,3][kink_save] = m[kink_save] - a[...,3][kink_save]
            d[...,3][kink_save] = 0.0
            b[...,3][kink_save] = n[kink_save]
            v[...,3][kink_save] = u(c[...,3][kink_save]) +\
                                  beta*solKinkSave['vFunc']
            vPm[...,3][kink_save] = uP(c[...,3][kink_save])
            vPn[...,3][kink_save] = r*beta*solKinkSave['vPnFunc']
            
        # Interior solution, partial liquid deposit, no liquid saving
        solCap = np.full_like(m,0.0)
        uPcap = np.full_like(m,0.0)
        dep = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            solCap[idx] = self.ConsIRAnext(yN[idx],r*(n[idx]+dMax))['vPnFunc']
            # if m <= dMax, will not reach illiquid savings cap
            uPcap[idx] = np.where(m[idx] > dMax,uP(m[idx] - dMax),np.inf)
            
            # neither bound is binding for deposits
            dep[idx] = ((uPdep[idx] < r*beta*solKink[idx]) & 
                        (uPcap[idx] > r*beta*solCap[idx]))
        
        if np.sum(dep) > 0: # if no one deposits, skip
            # loop through solutions for values of m,n,yN and create an array
            for idx in np.ndindex(m.shape):
                if dep[idx]:
                    d[...,4][idx] = br(self.dFOC,0.0,min(dMax,m[idx]),
                                       args=(m[idx],n[idx],yN[idx]))
        
            c[...,4][dep] = m[dep] - d[...,4][dep]
            a[...,4][dep] = 0.0
            b[...,4][dep] = n[dep] + d[...,4][dep]
            v[...,4][dep] = u(c[...,4][dep]) - k +\
                            beta*self.ConsIRAnext(yN[dep],
                                                  r*(n[dep] 
                                                  +d[...,4][dep]))['vFunc']
            vPm[...,4][dep] = uP(c[...,4][dep])
            vPn[...,4][dep] = uP(c[...,4][dep])
                    
        # Illiquid savings at cap, no liquid savings
        
        # upper bound on deposits and lower bound on liquid savings binds
        # cap on illiquid savings exceeds cash on hand
        solCapA = np.full_like(m,0.0)
        cap = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            solCapA[idx] = self.ConsIRAnext(yN[idx],r*(n[idx]+dMax))['vPmFunc']
            cap[idx] = ((uPcap[idx] <= r*beta*solCap[idx]) & 
                        (uPcap[idx] >= ra*beta*solCapA[idx]) &
                        (m[idx] > dMax))
        
        if np.sum(cap) > 0: # if no one is at cap w/ no liquid savings, skip
            c[...,5][cap] = m[cap] - dMax
            d[...,5][cap] = dMax
            a[...,5][cap] = 0.0
            b[...,5][cap] = n[cap] + dMax
            v[...,5][cap] = u(c[...,5][cap]) - k +\
                            beta*self.ConsIRAnext(yN[cap],r*(n[cap]+dMax)
                                                  )['vFunc']
            vPm[...,5][cap] = uP(c[...,5][cap])
            vPn[...,5][cap] = r*beta*solCap[cap]
            
        # Illiquid savings cap and liquid savings
        
        # upper bound on deposits binds and lower bound on liquid savings 
        # doesn't bind
        # cap on illiquid savings exceeds cash on hand
        cap_save = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            cap_save[idx] = ((uPcap[idx] <= r*beta*solCap[idx]) & 
                             (uPcap[idx] < ra*beta*solCapA[idx]) &
                             (m[idx] > dMax))
        
        
        if np.sum(cap_save) > 0: # if no one is at cap w/ liquid savings, skip
            # loop through solutions for values of m,n,yN and create an array
            for idx in np.ndindex(m.shape):
                if cap_save[idx]:
                    a[...,6][idx] = br(self.aFOC,0.0,m[idx]-dMax,
                                       args=(m[idx],n[idx],yN[idx]))
        
            solCapSave = self.ConsIRAnext(yN[cap_save]+ra*a[...,6][cap_save],
                                          r*(n[cap_save]+dMax))
        
            c[...,6][cap_save] = m[cap_save] - dMax - a[...,6][cap_save]
            d[...,6][cap_save] = dMax
            b[...,6][cap_save] = n[cap_save] + dMax
            v[...,6][cap_save] = u(c[...,6][cap_save]) - k +\
                                 beta*solCapSave['vFunc']
            vPm[...,6][cap_save] = uP(c[...,6][cap_save])
            vPn[...,6][cap_save] = r*beta*solCapSave['vPnFunc']
        
        # Find index of max utility among valid solutions
        max_state = np.argmax(v,axis=-1)
        
        # Create tuple of common dimensions for indexing max values
        max_dim = np.ogrid[[slice(i) for i in max_state.shape]]
        
        # Select elements from each array, based on index of max value
        c_star = c[tuple(max_dim) + (max_state,)]
        d_star = d[tuple(max_dim) + (max_state,)]
        a_star = a[tuple(max_dim) + (max_state,)]
        b_star = b[tuple(max_dim) + (max_state,)]
        v_star = v[tuple(max_dim) + (max_state,)]
        vPm_star = vPm[tuple(max_dim) + (max_state,)]
        vPn_star = vPn[tuple(max_dim) + (max_state,)]
        
        labDict = {0:'liq',1:'withdr',2:'kink',3:'kink_save',4:'dep',
                   5:'cap',6:'cap_save'}
        
        max_label = np.vectorize(lambda s: labDict[s])(max_state).tolist()
        
        solution = {'cFunc': c_star, 'dFunc': d_star, 'aFunc': a_star,
                    'bFunc': b_star, 'vFunc': v_star, 'vPmFunc': vPm_star,
                    'vPnFunc': vPn_star, 'max_state': max_state,
                    'max_label': max_label}
        
        if self.output == 'all':
            return solution
        else:
            return solution[self.output]

class ConsIRAPF(HARKobject):
    '''
    Solution for the IRA consumer with perfect foresight, after any initial
    period and before the terminal period.
    '''
    distance_criteria = ['period','NextIncome','DiscFac','CRRA','Rsave',
                         'Rira','PenIRA','MaxIRA','FixedCost','ConsIRAnext']
    
    def __init__(self,NextIncome,DiscFac,CRRA,Rsave,Rira,PenIRA,MaxIRA,
                 ConsIRAnext,FixedCost=0.0,output='all'):
        '''
        Constructor for solution in period after initial period and before
        the terminal period.
        
        Parameters
        ----------
        NextIncome : float or np.array
            Income in the next period.
        DiscFac : float
            Intertemporal discount factor for future utility.
        CRRA : float
            Coefficient of relative risk aversion.
        Rsave: float
            Interest factor on liquid assets between this period and the 
            succeeding period when assets are positive.
        Rira:  float
            Interest factor on illiquid assets between this period and the 
            succeeding period.
        PenIRA: float
            Penalty for early withdrawals (d < 0) from the illiqui account, 
            i.e. before t = T_ira.
        MaxIRA: float
            Maximum allowable IRA deposit, d <= MaxIRA
                FixedCost: float
        Fixed utility cost of making a withdrawal from the illiquid 
            account. 
        ConsIRAnext : function
            Returns optimal c,d,a and value function and marginal value
            function from the next period.
        output : string
            Whether consumption, deposit, or value function is output.
        
        Returns
        -------
        None
        '''
        
        self.NextIncome     = NextIncome
        self.DiscFac        = DiscFac
        self.CRRA           = CRRA
        self.Rsave          = Rsave
        self.Rira           = Rira
        self.PenIRA         = PenIRA
        self.MaxIRA         = MaxIRA
        self.FixedCost      = FixedCost
        self.yN             = NextIncome
        self.ConsIRAnext    = deepcopy(ConsIRAnext)
        self.output         = output
    
    @memoized
    def dFOC(self,d,m,n,yN):
        '''
        Evaluate expression for d, derived from the FOC for d in period 1 at
        an interior solution when making a deposit.
        
        Parameters
        ----------
        d : float or np.array
            Value of d, used to calculate value function next period.
        m : float or np.array
            Cash on hand, including period 1 income and liquid assets.
        n : float
            Illiquid account balance.
        yN : float or np.array
            Income next period
            
        Returns
        -------
        foc : float or np.array
            Derivative of the objective function at (d,m,n,yN).
        '''
        r = self.Rira
        beta = self.DiscFac
        t = self.PenIRA * (d < 0.0)
        ra = self.Rsave
        a = self.aOpt(d,m,n,yN)
        vPn = self.ConsIRAnext(yN + ra*a,r*(n + d))['vPnFunc']
        uP = (1.0-t)*utilityP(m - a - (1.0-t)*d,gam=self.CRRA)
        
        foc = uP - r*beta*vPn
        
        return foc
    
    @memoized
    def aFOC(self,a,d,m,n,yN):
        '''
        Evaluate the FOC for a, conditional on d.
        
        Parameters
        ----------
        a : float
            Liquid savings.
        d : float
            Deposit into or withdrawal from illiquid account.
        m : float
            Cash on hand, including period 1 income and liquid assets.
        n : float
            Illiquid account balance.
        yN : float
            Income next period
            
        Returns
        -------
        foc : float
            Derivative of the objective function at (a,m,n,yN).
        '''
        r = self.Rira
        ra = self.Rsave
        t = self.PenIRA * (d < 0.0)
        beta = self.DiscFac
        vPm = self.ConsIRAnext(yN + ra*a,r*(n+d))['vPmFunc']
        uP = utilityP(m - a - (1.0-t)*d,gam=self.CRRA)
        
        foc = uP - ra*beta*vPm
        
        return foc
    
    @memoized
    def aOpt(self,d,m,n,yN):
        '''
        Evaluate optimal liquid savings, conditional on the level of illiquid 
        deposit/withdrawal, liquid, and illiquid assets, and income next
        period.
        
        Parameters
        ----------
        d : float
            Deposit into or withdrawal from illiquid account.
        m : float or np.array
            Cash on hand, including period 1 income and liquid assets.
        n : float or np.array
            Illiquid account balance.
        yN : float or np.array
            Income next period
        
        Returns
        -------
        a_star : float
            Optimal liquid savings.
        '''
        beta = self.DiscFac
        r = self.Rira
        t = self.PenIRA * (d < 0.0)
        ra = self.Rsave
        uP = lambda c : utilityP(c,gam=self.CRRA) # marginal utility function
        
        # No liquid savings when making a withdrawal, taxes don't change, and
        # Rira > Rsave
        noSave = ((d < 0.0) & (self.PenIRA >= self.ConsIRAnext.PenIRA) &
                  (r > ra))
        
        if noSave:
            a_star = 0.0
            return a_star
        
        else:
            vPmCorner = self.ConsIRAnext(yN,r*(n + d))['vPmFunc']
            uPcorner = uP(m - (1.0-t)*d)
            
            # lower bound on liquid savings binds
            corner = ((noSave == False) & 
                      (uPcorner >= ra*beta*vPmCorner))
            
            if corner:
                a_star = 0.0
                return a_star
            
            else:
                a_star = br(self.aFOC,0.0,m - (1.0-t)*d,args=(d,m,n,yN))
                return a_star
            
    @memoized
    def __call__(self,m,n):
        '''
        Evaluate optimal consupmtion, deposit, savings, value and marginal
        value functions, given liquid and illiquid assets. If n and m don't
        have the same dimension, the one with the larger dimension takes
        precedent, and the first element of the one with the smaller dimension
        is used for all values of the dominent argument. If self.yN has a
        different dimension than the dominent argument, it is likewise reduced
        to its first element.
        
        Parameters
        ----------
        m : float or np.array
            Cash on hand, including current period income and liquid assets.
        n : float or np.array
            Illiquid account balance.
        
        Returns
        -------
        solution['cFunc'] : float or np.array
            Consumption in current period.
        solution['dFunc'] : float or np.array
            Withdrawal in current period.
        solution['aFunc'] : float or np.array
            Liquid assets at end of current period.
        solution['bFunc'] : float or np.array
            Illiquid assets at end of current period.
        solution['vFunc'] : float or np.array
            Value function in current period.
        solution['vPmFunc'] : float or np.array
            Marginal value function wrt m in current period.
        solution['vPnFunc'] : float or np.array
            Marginal value function wrt n in current period.
        '''
        # convert to np.arrays
        m = np.atleast_1d(m).astype(np.float)
        n = np.atleast_1d(n).astype(np.float)
        yN = np.atleast_1d(self.yN).astype(np.float)

        # Ensure comformability between m, n, and yN
        if m.shape != n.shape:
            if m.size >= n.size:
                n = np.full_like(m,n.item(0))
            else:
                m = np.full_like(n,m.item(0))
        
        if yN.shape != m.shape:
            yN = np.full_like(m,yN.item(0))
    
        beta = self.DiscFac
        r = self.Rira
        t = self.PenIRA
        ra = self.Rsave
        dMax = self.MaxIRA
        k = self.FixedCost
        u = lambda c : utility(c,gam=self.CRRA)  # utility function
        uP = lambda c : utilityP(c,gam=self.CRRA) # marginal utility function
        
        s = 5 # total possible states in this period
        
        # create placeholders with arrays of dimension s, for each element of m
        c = np.reshape(np.repeat(m,s,axis=-1),m.shape + (s,)) # consumption
        d = np.full_like(c,0.0) # deposit/withdrawal
        a = np.full_like(c,0.0) # liquid savings
        b = np.full_like(c,0.0) # illiquid savings
        v = np.full_like(c,-np.inf) # value function, initiated at -inf
        vPm = np.full_like(c,1.0) # marginal value wrt m
        vPn = np.full_like(c,1.0) # marginal value wrt n
               
        # Liquidate illiquid account, s = 0
        vPnLiq = np.full_like(m,0.0)
        uPliq = np.full_like(m,0.0)
        liq = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            a[...,0][idx] = self.aOpt(-n[idx],m[idx],n[idx],yN[idx]) # a at liq
            vPnLiq[idx] = self.ConsIRAnext(yN[idx] + ra*a[...,0][idx],
                                           0.0)['vPnFunc']
            uPliq[idx] = (1.0-t)*uP(m[idx] - a[...,0][idx] + (1.0-t)*n[idx])

            # lower bound on withdrawal is binding        
            liq[idx] = uPliq[idx] >= r*beta*vPnLiq[idx]
        
        if np.sum(liq) > 0: # if no one liquidates, skip
            c[...,0][liq] = m[liq] - a[...,0][liq] + (1.0-t)*n[liq]
            d[...,0][liq] = -n[liq]
            b[...,0][liq] = 0.0
            v[...,0][liq] = u(c[...,0][liq]) - k +\
                            beta*self.ConsIRAnext(yN[liq] + ra*a[...,0][liq],
                                                  np.zeros(n[liq].shape)
                                                  )['vFunc']
            vPm[...,0][liq] = uP(c[...,0][liq])
            vPn[...,0][liq] = (1.0-t)*uP(c[...,0][liq])       
            
        # Interior solution, partial illiquid withdrawal, s = 1
        vPnKink = np.full_like(m,0.0)
        uPwithdr = np.full_like(m,0.0)
        withdr = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            a[...,2][idx] = self.aOpt(0.0,m[idx],n[idx],yN[idx]) # a at kink
            vPnKink[idx] = self.ConsIRAnext(yN[idx] + ra*a[...,2][idx],
                                            r*n[idx])['vPnFunc']
            uPwithdr[idx] = (1.0-t)*uP(m[idx] - a[...,1][idx])
        
            # neither bound on withdrawals binds
            withdr[idx] = ((uPliq[idx] < r*beta*vPnLiq[idx]) &
                           (uPwithdr[idx] > r*beta*vPnKink[idx]))
        
        # interior solution for withdrawal
        if np.sum(withdr) > 0: # if no one withdraws, skip
            # loop through solutions for values of m,n,yN and create an array
            for idx in np.ndindex(m.shape):
                if withdr[idx]:
                    d[...,1][idx] = br(self.dFOC,-n[idx],0.0,
                                       args=(m[idx],n[idx],yN[idx]))
                    a[...,1][idx] = self.aOpt(d[...,1][idx],m[idx],n[idx],
                                              yN[idx])
            
            c[...,1][withdr] = m[withdr] - a[...,1][withdr] -\
                               (1.0-t)*d[...,1][withdr]
            b[...,1][withdr] = n[withdr] + d[...,1][withdr]
            v[...,1][withdr] = u(c[...,1][withdr]) - k +\
                               beta*self.ConsIRAnext(yN[withdr] +\
                                                     ra*a[...,1][withdr],
                                                     r*(n[withdr]
                                                        +d[...,1][withdr])
                                                     )['vFunc']
            vPm[...,1][withdr] = uP(c[...,1][withdr])
            vPn[...,1][withdr] = (1.0-t)*uP(c[...,1][withdr])
        
        # Corner solution w/ no illiquid withdrawal or saving, s = 2
        # Calculate for everyone
        c[...,2] = m - a[...,2]
        d[...,2] = 0.0
        b[...,2] = n
        v[...,2] = u(c[...,2]) + beta*self.ConsIRAnext(yN + ra*a[...,2]
                                                       ,r*n)['vFunc']
        vPm[...,2] = uP(c[...,2])
        vPn[...,2] = r*beta*vPnKink
            
        # Interior solution, partial liquid deposit, s = 3
        uPdep = np.full_like(m,0.0)        
        vPnCap = np.full_like(m,0.0)
        uPcap = np.full_like(m,0.0)
        dep = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            uPdep[idx] = uP(m[idx] - a[...,2][idx])
            # if m <= dMax, will not reach illiquid savings cap
            if m[idx] > dMax:
                a[...,4][idx] = self.aOpt(dMax,m[idx],n[idx],yN[idx])# a at cap
                vPnCap[idx] = self.ConsIRAnext(yN[idx] + ra*a[...,4][idx],
                                               r*(n[idx] + dMax))['vPnFunc']
                uPcap[idx] = uP(m[idx] - a[...,4][idx] - dMax)
            else:
                vPnCap[idx] = -np.inf
                uPcap[idx] = np.inf
            
            # neither bound is binding for deposits
            dep[idx] = ((uPdep[idx] < r*beta*vPnKink[idx]) &
                        (uPcap[idx] > r*beta*vPnCap[idx]))
        
        if np.sum(dep) > 0: # if no one deposits, skip
            # loop through solutions for values of m,n,yN and create an array
            for idx in np.ndindex(m.shape):
                if dep[idx]:
                    d[...,3][idx] = br(self.dFOC,0.0,min(dMax,m[idx]),
                                       args=(m[idx],n[idx],yN[idx]))
                    a[...,3][idx] = self.aOpt(d[...,3][idx],m[idx],n[idx],
                                              yN[idx])
        
            c[...,3][dep] = m[dep] - a[...,3][dep] - d[...,3][dep]
            b[...,3][dep] = n[dep] + d[...,3][dep]
            v[...,3][dep] = u(c[...,3][dep]) - k +\
                            beta*self.ConsIRAnext(yN[dep] + ra*a[...,3][dep],
                                                  r*(n[dep]
                                                  +d[...,3][dep]))['vFunc']
            vPm[...,3][dep] = uP(c[...,3][dep])
            vPn[...,3][dep] = uP(c[...,3][dep])
                    
        # Illiquid savings at cap, s = 4
        
        # upper bound on deposits and lower bound on liquid savings binds
        # cap on illiquid savings exceeds cash on hand
        cap = np.full_like(m,False,dtype='bool')
        
        for idx in np.ndindex(m.shape):
            cap[idx] = ((uPcap[idx] <= r*beta*vPnCap[idx]) &
                        (m[idx] > dMax))
        
        if np.sum(cap) > 0: # if no one is at cap, skip
            c[...,4][cap] = m[cap] - a[...,4][cap] - dMax
            d[...,4][cap] = dMax
            b[...,4][cap] = n[cap] + dMax
            v[...,4][cap] = u(c[...,4][cap]) - k +\
                            beta*self.ConsIRAnext(yN[cap] + ra*a[...,4][cap],
                                                  r*(n[cap] + dMax))['vFunc']
            vPm[...,4][cap] = uP(c[...,4][cap])
            vPn[...,4][cap] = r*beta*vPnCap[cap]
        
        # Find index of max utility among valid solutions
        max_state = np.argmax(v,axis=-1)
        
        # Create tuple of common dimensions for indexing max values
        max_dim = np.ogrid[[slice(i) for i in max_state.shape]]
        
        # Select elements from each array, based on index of max value
        c_star = c[tuple(max_dim) + (max_state,)]
        d_star = d[tuple(max_dim) + (max_state,)]
        a_star = a[tuple(max_dim) + (max_state,)]
        b_star = b[tuple(max_dim) + (max_state,)]
        v_star = v[tuple(max_dim) + (max_state,)]
        vPm_star = vPm[tuple(max_dim) + (max_state,)]
        vPn_star = vPn[tuple(max_dim) + (max_state,)]
        
        labDict = {0:'liq',1:'withdr',2:'kink',3:'dep',4:'cap'}
        
        max_label = np.vectorize(lambda s,a: labDict[s] if a == 0.0 
                                 else labDict[s] + '_save')(max_state,
                                                           a_star).tolist()
        
        solution = {'cFunc': c_star, 'dFunc': d_star, 'aFunc': a_star,
                    'bFunc': b_star, 'vFunc': v_star, 'vPmFunc': vPm_star,
                    'vPnFunc': vPn_star, 'max_state': max_state,
                    'max_label': max_label}
        
        if self.output == 'all':
            return solution
        else:
            return solution[self.output]    
    
class ConsIRAPFinitial(HARKobject):
    '''
    Solution for IRA consumer model with perfect foresight, in initial period
    where agent doesn't consume, but only allocates resources to illiquid
    and liquid accounts.
    '''
    distance_criteria = ['NextIncome','DiscFac','CRRA','Rsave',
                         'Rira','ConsIRAnext']
    
    def __init__(self,NextIncome,DiscFac,CRRA,Rsave,Rira,ConsIRAnext,
                 output='all'):
        '''
        Constructor for period 0 solution.
        
        Parameters
        ----------
        NextIncome : float or np.array
            Income in the next period.
        DiscFac : float
            Intertemporal discount factor for future utility.
        CRRA : float
            Coefficient of relative risk aversion.
        Rsave: float
            Interest factor on liquid assets between this period and the 
            succeeding period when assets are positive.
        Rira:  float
            Interest factor on illiquid assets between this period and the 
            succeeding period.
        ConsIRAnext : function
            Returns optimal c,d,a and value function and marginal value
            function from the next period.
        output : string
            Whether consumption, deposit, or value function is output.
        
        Returns
        -------
        None
        '''
        
        self.NextIncome     = NextIncome
        self.DiscFac        = DiscFac
        self.CRRA           = CRRA
        self.Rsave          = Rsave
        self.Rira           = Rira
        self.yN             = NextIncome
        self.ConsIRAnext    = deepcopy(ConsIRAnext)
        self.output         = output
    
    @memoized    
    def aFOC(self,a,w,yN):
        '''
        Evaluate expression for the FOC for a in initial period. Satisfies the 
        budget constraint: w = a + d.
        
        Parameters
        ----------
        a : float or np.array
            Liquid savings in period 0.
        w : float or np.array
            Initial assets in period 0.
        yN : float or np.array
            Income next period
            
        Returns
        -------
        astar : float or np.array
            FOC for a at an interior solution.
        '''
        r = self.Rira
        ra = self.Rsave
        solN = self.ConsIRAnext(yN + ra*a,r*(w - a))
        
        foc = ra*solN['vPmFunc'] - r*solN['vPnFunc']
        
        return foc
    
    @memoized
    def __call__(self,w):
        '''
        Evaluate optimal initial asset allocation in the initial period.
        
        Parameters
        ----------
        w : float or np.array
            Initial assets in.
        
        Returns
        -------
        solution['cFunc'] : float or np.array
            Consumption in current period.
        solution['dFunc'] : float or np.array
            Withdrawal in current period.
        solution['aFunc'] : float or np.array
            Liquid assets at end of current period.
        solution['bFunc'] : float or np.array
            Illiquid assets at end of current period.
        solution['vFunc'] : float or np.array
            Value function in current period.
        solution['vPmFunc'] : float or np.array
            Marginal value function wrt m in current period.
        solution['vPnFunc'] : float or np.array
            Marginal value function wrt n in current period.
        '''
        # convert to np.arrays
        w = np.atleast_1d(w).astype(np.float)
        yN = np.atleast_1d(self.yN).astype(np.float)
        
        # Ensure comformability between w and yN        
        if yN.shape != w.shape:
            yN = np.full_like(w,yN.item(0))
        
        beta = self.DiscFac
        r = self.Rira
        ra = self.Rsave
        
        s = 3 # total possible states in this period
        
        # create placeholders with arrays of dimension s, for each element of w
        a = np.reshape(np.repeat(w,s,axis=-1),w.shape + (s,)) # liquid savings
        c = np.full_like(a,0.0) # consumption
        d = np.full_like(a,0.0) # deposit/withdrawal
        b = np.full_like(a,0.0) # illiquid assets
        v = np.full_like(a,-np.inf) # value function, initiated at -inf
        vPm = np.full_like(a,1.0) # marginal value wrt m
        vPn = np.full_like(a,0.0) # marginal value wrt n
        
        # Corner solution with all assets placed in liquid account, s = 0
        solLiq = self.ConsIRAnext(yN + ra*w,np.zeros(w.shape))
        liq = np.full_like(w,False,dtype='bool')
        
        for idx in np.ndindex(w.shape):
            liq[idx] = (ra*solLiq['vPmFunc'][idx] >= r*solLiq['vPnFunc'][idx])
        
        if np.sum(liq) > 0: # if no one liquidates, skip
            c[...,0][liq] = 0.0
            d[...,0][liq] = 0.0
            a[...,0][liq] = w[liq]
            b[...,0][liq] = 0.0
            v[...,0][liq] = beta*solLiq['vFunc'][liq]
            vPm[...,0][liq] = ra*beta*solLiq['vPmFunc'][liq]
            vPn[...,0][liq] = 0.0
        
        # Interior solution with positive liquid and illiquid savings, s = 1
        solCap = self.ConsIRAnext(yN,r*w)
        inter = np.full_like(w,False,dtype='bool')
        
        for idx in np.ndindex(w.shape):
            inter[idx] = ((ra*solLiq['vPmFunc'][idx] < 
                           r*solLiq['vPnFunc'][idx]) &
                          (ra*solCap['vPmFunc'][idx] > 
                           r*solCap['vPnFunc'][idx]))
        
        if np.sum(inter) > 0: # if no one is at interior, skip
            c[...,1][inter] = 0.0
            
            # loop through solutions for values of m,n,yN and create an array
            for idx in np.ndindex(w.shape):
                if inter[idx]:
                    a[...,1][idx] = br(self.aFOC,0.0,w[idx],
                                       args=(w[idx],yN[idx]))

            d[...,1][inter] = w[inter] - a[...,1][inter]
            b[...,1][inter] = d[...,1][inter]
        
            solInter = self.ConsIRAnext(yN[inter] + ra*a[...,1][inter],
                                        r*(d[...,1][inter]))
        
            v[...,1][inter] = beta*solInter['vFunc']
            vPm[...,1][inter] = ra*beta*solInter['vPmFunc']
            vPn[...,1][inter] = 0.0
        
        # Corner solution with all asstes placed in illiquid account, s = 2
        cap = np.full_like(w,False,dtype='bool')
        
        for idx in np.ndindex(w.shape):
            cap[idx] = ra*solCap['vPmFunc'][idx] <= r*solCap['vPnFunc'][idx]
        
        if np.sum(cap) > 0: # if no one is at cap, skip
            c[...,2][cap] = 0.0
            a[...,2][cap] = 0.0
            d[...,2][cap] = w[cap]
            b[...,2][cap] = w[cap]
            v[...,2][cap] = beta*solCap['vFunc'][cap]
            vPm[...,1][cap] = r*beta*solCap['vPmFunc'][cap]
            vPn[...,1][cap] = 0.0

        # Find index of max utility among valid solutions
        max_state = np.argmax(v,axis=-1)
        
        # Create tuple of common dimensions for indexing max values
        max_dim = np.ogrid[[slice(i) for i in max_state.shape]]
        
        # Select elements from each array, based on index of max value
        c_star = c[tuple(max_dim) + (max_state,)]
        d_star = d[tuple(max_dim) + (max_state,)]
        a_star = a[tuple(max_dim) + (max_state,)]
        b_star = b[tuple(max_dim) + (max_state,)]
        v_star = v[tuple(max_dim) + (max_state,)]
        vPm_star = vPm[tuple(max_dim) + (max_state,)]
        vPn_star = vPn[tuple(max_dim) + (max_state,)]
        
        labDict = {0:'liq',1:'inter',2:'cap'}
        
        max_label = np.vectorize(lambda s: labDict[s])(max_state).tolist()
        
        solution = {'cFunc': c_star, 'dFunc': d_star, 'aFunc': a_star,
                    'bFunc': b_star, 'vFunc': v_star, 'vPmFunc': vPm_star,
                    'vPnFunc': vPn_star, 'max_state': max_state,
                    'max_label': max_label}
        
        if self.output == 'all':
            return solution
        else:
            return solution[self.output]

# ====================================
# === Perfect foresight IRA model ===
# ====================================
        
class IRAPerfForesightConsumerType(HARKobject):
    '''
    Not the final product: short term consumer class for solving a lifecycle
    consumer model with an illiquid IRA and a liquid account, with no
    borrowing, and expiration of the penalty after a given period.
    '''
    def __init__(self,IncomeProfile,DiscFac,CRRA,Rsave,Rira,PenIRA,MaxIRA,
                 FixedCost,T_cycle,T_ira,InitialProblem):
        
        self.IncomeProfile      = IncomeProfile
        self.IncomeProfile0     = deepcopy(IncomeProfile)
        self.DiscFac            = DiscFac
        self.CRRA               = CRRA
        self.Rsave              = Rsave
        self.Rira               = Rira
        self.PenIRA             = PenIRA
        self.MaxIRA             = MaxIRA
        self.FixedCost          = FixedCost
        self.T_cycle            = T_cycle
        self.T_ira              = T_ira
        self.InitialProblem     = InitialProblem
    
    def solve(self):
        
        # Initialize the solution
        PenT = 0.0 if self.T_ira <= self.T_cycle else self.PenIRA
        
        solution = [ConsIRAPFterminal(CRRA=self.CRRA,PenIRA=PenT,
                                       FixedCost=self.FixedCost)]
        
        # Add rest of solutions
        for i in reversed(range(self.T_cycle-1)):
            if i >= self.T_ira - 1:
                solution.append(ConsIRAPF(
                                          NextIncome=self.IncomeProfile[i + 1],
                                          DiscFac=self.DiscFac, CRRA=self.CRRA,
                                          Rsave=self.Rsave, Rira=self.Rira,
                                          PenIRA=0.0, MaxIRA=self.MaxIRA,
                                   ConsIRAnext=solution[self.T_cycle - i - 2],
                                          FixedCost=self.FixedCost))
            
            elif ((i < self.T_ira - 1 and i > 0) or 
                  (i == 0 and not self.InitialProblem)):
                solution.append(ConsIRAPF(
                                          NextIncome=self.IncomeProfile[i + 1],
                                          DiscFac=self.DiscFac, CRRA=self.CRRA,
                                          Rsave=self.Rsave, Rira=self.Rira,
                                          PenIRA=self.PenIRA, 
                                          MaxIRA=self.MaxIRA,
                                   ConsIRAnext=solution[self.T_cycle - i - 2],
                                          FixedCost=self.FixedCost))
            
            else:
                solution.append(ConsIRAPFinitial(
                                          NextIncome=self.IncomeProfile[i + 1],
                                                 DiscFac=self.DiscFac,
                                                 CRRA=self.CRRA,
                                                 Rsave=self.Rsave,
                                                 Rira=self.Rira,
                                   ConsIRAnext=solution[self.T_cycle - i - 2]))

        solution.reverse()        
        
        self.solution = solution
    
    def simulate(self,w0):
        
        self.w0 = w0
        self.IncomeProfile = copy(self.IncomeProfile0)
        self.solve()
        
        pt = progress_timer(description= 'Simulating Lifecycle',
                        n_iter=self.T_cycle)
        
        pt.update()
        
        if self.InitialProblem:
            simulation = [self.solution[0](w0)]
        
        else:
            simulation = [self.solution[0](w0,0.0)]
            
        for i in range(1,self.T_cycle):
            
            pt.update()
            
            simulation.append(
                          self.solution[i](self.IncomeProfile[i] + 
                                           self.Rsave*simulation[i-1]['aFunc'],
                                           self.Rira*simulation[i-1]['bFunc']))
        
        self.simulation = simulation
        
        pt.finish()
        
    def simulate1Dip(self,w0,tDip,dipSize):
        
        self.w0 = w0
        self.IncomeProfile = copy(self.IncomeProfile0)
        self.IncomeProfile[tDip] *= dipSize
        self.solve()
        
        pt = progress_timer(description= 'Simulating Lifecycle',
                        n_iter=self.T_cycle)
        
        pt.update()
        
        if self.InitialProblem:
            simulation = [self.solution[0](w0)]
        
        else:
            simulation = [self.solution[0](w0,0.0)]
            
        for i in range(1,self.T_cycle):
            
            pt.update()
            
            simulation.append(
                          self.solution[i](self.IncomeProfile[i] + 
                                           self.Rsave*simulation[i-1]['aFunc'],
                                           self.Rira*simulation[i-1]['bFunc']))
        
        self.simulation = simulation
        
        pt.finish()
        
    def simulate2Dip(self,w0,tDip,dipSize):
        
        self.w0 = w0
        self.IncomeProfile = copy(self.IncomeProfile0)
        self.IncomeProfile[tDip:tDip + 2] *= dipSize
        self.solve()
        
        pt = progress_timer(description= 'Simulating Lifecycle',
                        n_iter=self.T_cycle)
        
        pt.update()
        
        if self.InitialProblem:
            simulation = [self.solution[0](w0)]
        
        else:
            simulation = [self.solution[0](w0,0.0)]
            
        for i in range(1,self.T_cycle):
            
            pt.update()
            
            simulation.append(
                          self.solution[i](self.IncomeProfile[i] + 
                                           self.Rsave*simulation[i-1]['aFunc'],
                                           self.Rira*simulation[i-1]['bFunc']))
        
        self.simulation = simulation
        
        pt.finish()
        
    def simulatePDip(self,w0,tDip,dipSize):
        
        self.w0 = w0
        self.IncomeProfile = copy(self.IncomeProfile0)
        a = (1.0 - dipSize)*\
            np.sum(self.IncomeProfile[tDip:tDip+2])/\
            np.sum(self.IncomeProfile[tDip:])
        self.IncomeProfile[tDip:] *= 1 - a
        self.solve()
        
        pt = progress_timer(description= 'Simulating Lifecycle',
                        n_iter=self.T_cycle)
        
        pt.update()
        
        if self.InitialProblem:
            simulation = [self.solution[0](w0)]
        
        else:
            simulation = [self.solution[0](w0,0.0)]
            
        for i in range(1,self.T_cycle):
            
            pt.update()
            
            simulation.append(
                          self.solution[i](self.IncomeProfile[i] + 
                                           self.Rsave*simulation[i-1]['aFunc'],
                                           self.Rira*simulation[i-1]['bFunc']))
        
        self.simulation = simulation
        
        pt.finish()
        
    def simulate1Shock(self,w0,tDip,dipSize):
        
        self.w0 = w0
        self.IncomeProfile = copy(self.IncomeProfile0)
        self.solve()
        
        pt = progress_timer(description= 'Simulating Lifecycle',
                        n_iter=self.T_cycle)
        
        pt.update()
        
        if self.InitialProblem:
            simulation = [self.solution[0](w0)]
        
        else:
            simulation = [self.solution[0](w0,0.0)]

        for i in range(1,tDip):
            
            pt.update()
            
            simulation.append(
                        self.solution[i](self.IncomeProfile[i] + 
                                           self.Rsave*simulation[i-1]['aFunc'],
                                           self.Rira*simulation[i-1]['bFunc']))
        
        self.oldSolution = copy(self.solution)
        
        self.IncomeProfile[tDip] *= dipSize
        self.solve()

        for i in range(tDip,self.T_cycle):
            
            pt.update()
            
            simulation.append(
                        self.solution[i](self.IncomeProfile[i] + 
                                           self.Rsave*simulation[i-1]['aFunc'],
                                           self.Rira*simulation[i-1]['bFunc']))
        
        self.simulation = simulation
        self.solution[:tDip] = self.oldSolution[:tDip]
        
        pt.finish()
        
    def simulate2Shock(self,w0,tDip,dipSize):
        
        self.w0 = w0
        self.IncomeProfile = copy(self.IncomeProfile0)
        self.solve()
        
        pt = progress_timer(description= 'Simulating Lifecycle',
                        n_iter=self.T_cycle)
        
        pt.update()
        
        if self.InitialProblem:
            simulation = [self.solution[0](w0)]
        
        else:
            simulation = [self.solution[0](w0,0.0)]

        for i in range(1,tDip):
            
            pt.update()
            
            simulation.append(
                        self.solution[i](self.IncomeProfile[i] + 
                                           self.Rsave*simulation[i-1]['aFunc'],
                                           self.Rira*simulation[i-1]['bFunc']))
        
        self.oldSolution = copy(self.solution)
        
        self.IncomeProfile[tDip:tDip + 2] *= dipSize
        self.solve()
        
        for i in range(tDip,self.T_cycle):
            
            pt.update()
            
            simulation.append(
                        self.solution[i](self.IncomeProfile[i] + 
                                           self.Rsave*simulation[i-1]['aFunc'],
                                           self.Rira*simulation[i-1]['bFunc']))
        
        self.simulation = simulation
        self.solution[:tDip] = self.oldSolution[:tDip]
        
        pt.finish()
        
    def simulatePShock(self,w0,tDip,dipSize):
        
        self.w0 = w0
        self.IncomeProfile = copy(self.IncomeProfile0)
        self.solve()
        
        pt = progress_timer(description= 'Simulating Lifecycle',
                        n_iter=self.T_cycle)
        
        pt.update()
        
        if self.InitialProblem:
            simulation = [self.solution[0](w0)]
        
        else:
            simulation = [self.solution[0](w0,0.0)]

        for i in range(1,tDip):
            
            pt.update()
            
            simulation.append(
                        self.solution[i](self.IncomeProfile[i] + 
                                           self.Rsave*simulation[i-1]['aFunc'],
                                           self.Rira*simulation[i-1]['bFunc']))
        
        oldSolution = copy(self.solution)
        
        a = (1.0 - dipSize)*\
            np.sum(self.IncomeProfile[tDip:tDip+2])/\
            np.sum(self.IncomeProfile[tDip:])
        self.IncomeProfile[tDip:] *= 1 - a
        self.solve()
        
        for i in range(tDip,self.T_cycle):
            
            pt.update()
            
            simulation.append(
                        self.solution[i](self.IncomeProfile[i] + 
                                           self.Rsave*simulation[i-1]['aFunc'],
                                           self.Rira*simulation[i-1]['bFunc']))
        
        self.simulation = simulation
        self.solution[:tDip] = oldSolution[:tDip]
        
        pt.finish()
        
    def graphSim(self,saveFig=0,savePath='',graphLab =''):
        
        # create lifecycle arrays
        keys = ['aFunc','bFunc','cFunc','dFunc']
        
        a, b, c, d = [np.concatenate([k[ki] for k in self.simulation],axis=0) 
                                      for ki in keys]
        
        y = self.IncomeProfile
        
        dep = np.maximum(d,0)
        withdr = -np.minimum(d,0)
        
        # Plot Assets
        tvar = np.arange(self.T_cycle)
        plt.plot(tvar,b,'C1',label='Illiquid Assets')
        plt.plot(tvar,a,'C0--',label='Liquid Assets')
        plt.xlabel('Time')
        plt.ylabel('Balance')
        plt.title('Asset Accumulation, beta=' + str(self.DiscFac) +', w0=' 
                  + str(self.w0) + ', t=' + str(self.PenIRA))
        plt.legend()
        plt.grid()
        plt.xticks(tvar)
        plt.axvline(x=self.T_ira-1,color = 'C3')
        if saveFig:
            plt.savefig(savePath + '/IRAPFassets_' + graphLab + '.png')
        plt.show()
        
        # Plot Withdrawals and Deposits
        plt.plot(tvar,dep,'C1',label='Deposits')
        plt.plot(tvar,withdr,'C0--',label='Withdrawals')
        plt.xlabel('Time')
        plt.ylabel('Deposits/Withdrawals')
        plt.title('Deposits/Withdrawals, beta=' + str(self.DiscFac) +', w0=' 
                  + str(self.w0) + ', t=' + str(self.PenIRA))
        plt.legend()
        plt.grid()
        plt.xticks(tvar)
        plt.axvline(x=self.T_ira-1,color = 'C3')
        if saveFig:
            plt.savefig(savePath + '/IRAPFwithdr_' + graphLab + '.png')
        plt.show()
        
        #Plot Consumption and Income
        plt.plot(tvar[1:],c[1:],'C1',label='Consumption')
        plt.plot(tvar[1:],y[1:],'C0--',label='Income')
        plt.xlabel('Time')
        plt.ylabel('Consumption/Income')
        plt.title('Income/Consumption, beta=' + str(self.DiscFac) +', w0=' 
                  + str(self.w0) + ', t=' + str(self.PenIRA))
        plt.legend()
        plt.grid()
        plt.xticks(tvar)
        plt.locator_params(axis='y', nbins=6)
        plt.axvline(x=self.T_ira-1,color = 'C3')
        if saveFig:
            plt.savefig(savePath + '/IRAPFcons_' + graphLab + '.png')
        plt.show()
        
###############################################################################

def main():
    
    w0 = 0.25
    T = 8
    T_ira = 6
    y = np.array(T*[1.0])
    beta = 0.95
    g = 2
    ra = 1
    r = 1.1
    dMax = .5
    t = .2
    k = 0.0
    simulations = {}
    
    IRAPF = IRAPerfForesightConsumerType(y,beta,g,ra,r,t,dMax,k,T,T_ira,1)
    IRAPF.solve()
    IRAPF.simulate(w0)
    IRAPF.graphSim(saveFig=1,savePath='IRA_Results2',graphLab='8p')
    simulations['8p'] = IRAPF.simulation
    
    
    for i in range(1,T-1):
        IRAPF.simulate1Dip(w0,i,.75)
        simulations['8p' + str(i)] = IRAPF.simulation
        IRAPF.graphSim(saveFig=1,savePath='IRA_Results2',graphLab='8p' 
                       + str(i))
        IRAPF.simulate2Dip(w0,i,.75)
        simulations['8p' + str(i) + 's'] = IRAPF.simulation
        IRAPF.graphSim(saveFig=1,savePath='IRA_Results2',graphLab='8p' 
                       + str(i) + 's')
        IRAPF.simulatePDip(w0,i,.75)
        simulations['8p' + str(i) + 'p'] = IRAPF.simulation
        IRAPF.graphSim(saveFig=1,savePath='IRA_Results2',graphLab='8p' 
                       + str(i) + 'p')
        
    with open('IRA_Results2/IRAPF_Simulations.pickle','wb') as handle:
        pickle.dump(simulations, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('IRA_Results2/IRAPF_Simulations.pickle', 'rb') as handle:
        stored_simulation = pickle.load(handle)
  
#    
#    # Permanent dip in income
#    
#    y[1:] = 1 - .5/y[1:].size
#    IRAPF = IRAPerfForesightConsumerType(y,beta,g,ra,r,t,dMax,T,T_ira,1)
#    IRAPF.solve()
#    IRAPF.simulate(w0)
#    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P1p')
#    
#    simulations['8P1p'] = IRAPF.simulation
#    
#    y[1] = 1
#    y[2:] = 1 - .5/y[2:].size
#    IRAPF = IRAPerfForesightConsumerType(y,beta,g,ra,r,t,dMax,T,T_ira,1)
#    IRAPF.solve()
#    IRAPF.simulate(w0)
#    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P2p')
#    
#    simulations['8P2p'] = IRAPF.simulation
#    
#    y[2] = 1
#    y[3:] = 1 - .5/y[3:].size
#    IRAPF = IRAPerfForesightConsumerType(y,beta,g,ra,r,t,dMax,T,T_ira,1)
#    IRAPF.solve()
#    IRAPF.simulate(w0)
#    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P3p')
#    
#    simulations['8P3p'] = IRAPF.simulation
#    
#    y[3] = 1
#    y[4:] = 1 - .5/y[4:].size
#    IRAPF = IRAPerfForesightConsumerType(y,beta,g,ra,r,t,dMax,T,T_ira,1)
#    IRAPF.solve()
#    IRAPF.simulate(w0)
#    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P4p')
#    
#    simulations['8P4p'] = IRAPF.simulation
#    
#    y[4] = 1
#    y[5:] = 1 - .5/y[5:].size
#    IRAPF = IRAPerfForesightConsumerType(y,beta,g,ra,r,t,dMax,T,T_ira,1)
#    IRAPF.solve()
#    IRAPF.simulate(w0)
#    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P5p')
#    
#    simulations['8P5p'] = IRAPF.simulation
#    
#    
#    
#    with open('IRA_Results/IRAPF_Simulations.pickle', 'rb') as handle:
#        simulations = pickle.load(handle)
#    
#    IRAPF = IRAPerfForesightConsumerType(y,beta,g,ra,r,t,dMax,T,T_ira,1)
#    IRAPF.solve()
#    IRAPF.w0 = w0
#    IRAPF.simulation = simulations['8P']
##    IRAPF.simulate(w0)
#    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P')
#    
#    simulations['8P'] = IRAPF.simulation
#    
#    # Single period dips
#    
#    y[1] = .75
#    
#    IRAPF = IRAPerfForesightConsumerType(y,beta,g,ra,r,t,dMax,T,T_ira,1)
#    IRAPF.solve()
#    IRAPF.w0 = w0
#    IRAPF.simulation = simulations['8P1']
##    IRAPF.simulate(w0)
#    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P1')
#    
#    simulations['8P1'] = IRAPF.simulation
#
#    y[1] = 1.0
#    y[2] = .75
#    
#    IRAPF = IRAPerfForesightConsumerType(y,beta,g,ra,r,t,dMax,T,T_ira,1)
#    IRAPF.solve()
#    IRAPF.w0 = w0
#    IRAPF.simulation = simulations['8P2']
##    IRAPF.simulate(w0)
#    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P2')
#    
#    simulations['8P2'] = IRAPF.simulation
#    
#    y[2] = 1.0
#    y[3] = .75
#    
#    IRAPF = IRAPerfForesightConsumerType(y,beta,g,ra,r,t,dMax,T,T_ira,1)
#    IRAPF.solve()
#    IRAPF.w0 = w0
#    IRAPF.simulation = simulations['8P3']
##    IRAPF.simulate(w0)
#    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P3')
#    
#    simulations['8P3'] = IRAPF.simulation
#    
#    y[3] = 1.0
#    y[4] = .75
#    
#    IRAPF = IRAPerfForesightConsumerType(y,beta,g,ra,r,t,dMax,T,T_ira,1)
#    IRAPF.solve()
#    IRAPF.w0 = w0
#    IRAPF.simulation = simulations['8P4']
##    IRAPF.simulate(w0)
#    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P4')
#    
#    simulations['8P4'] = IRAPF.simulation
#    
#    y[4] = 1.0
#    y[5] = .75
#    
#    IRAPF = IRAPerfForesightConsumerType(y,beta,g,ra,r,t,dMax,T,T_ira,1)
#    IRAPF.solve()
#    IRAPF.w0 = w0
#    IRAPF.simulation = simulations['8P5']
##    IRAPF.simulate(w0)
#    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P5')
#    
#    simulations['8P5'] = IRAPF.simulation
#    
#    # Serially correlated dips
#    
#    y = np.array(T*[1.0])
#    y[1] = .75
#    y[2] = .75
#    
#    IRAPF = IRAPerfForesightConsumerType(y,beta,g,ra,r,t,dMax,T,T_ira,1)
#    IRAPF.solve()
#    IRAPF.w0 = w0
#    IRAPF.simulation = simulations['8P1s']
##    IRAPF.simulate(w0)
#    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P1s')
#    
#    simulations['8P1s'] = IRAPF.simulation
#
#    
#    y[1] = 1.0
#    y[3] = .75
#    
#    IRAPF = IRAPerfForesightConsumerType(y,beta,g,ra,r,t,dMax,T,T_ira,1)
#    IRAPF.solve()
#    IRAPF.w0 = w0
#    IRAPF.simulation = simulations['8P2s']
##    IRAPF.simulate(w0)
#    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P2s')
#    
#    simulations['8P2s'] = IRAPF.simulation
#    
#    y[2] = 1.0
#    y[4] = .75
#    
#    IRAPF = IRAPerfForesightConsumerType(y,beta,g,ra,r,t,dMax,T,T_ira,1)
#    IRAPF.solve()
#    IRAPF.w0 = w0
#    IRAPF.simulation = simulations['8P3s']
##    IRAPF.simulate(w0)
#    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P3s')
#    
#    simulations['8P3s'] = IRAPF.simulation
#    
#    y[3] = 1.0
#    y[5] = .75
#    
#    IRAPF = IRAPerfForesightConsumerType(y,beta,g,ra,r,t,dMax,T,T_ira,1)
#    IRAPF.solve()
#    IRAPF.w0 = w0
#    IRAPF.simulation = simulations['8P4s']
##    IRAPF.simulate(w0)
#    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P4s')
#    
#    simulations['8P4s'] = IRAPF.simulation
#    
#    y[4] = 1.0
#    y[6] = .75
#    
#    IRAPF = IRAPerfForesightConsumerType(y,beta,g,ra,r,t,dMax,T,T_ira,1)
#    IRAPF.solve()
#    IRAPF.w0 = w0
#    IRAPF.simulation = simulations['8P5s']
##    IRAPF.simulate(w0)
#    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P5s')   
#    
#    simulations['8P5s'] = IRAPF.simulation
#    
#    with open('IRA_Results/IRAPF_Simulations2.pickle','wb') as handle:
#        pickle.dump(simulations, handle, protocol=pickle.HIGHEST_PROTOCOL)
#        
#    with open('IRA_Results/IRAPF_Simulations2.pickle', 'rb') as handle:
#        stored_simulation = pickle.load(handle)
#        
if __name__ == '__main__':
    main()