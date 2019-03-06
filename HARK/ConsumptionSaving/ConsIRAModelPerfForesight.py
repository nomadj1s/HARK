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
from scipy.optimize import basinhopping
from scipy.optimize import brentq as br
from time import clock, time
import multiprocessing as mp
from pathos.multiprocessing import ProcessPool
import itertools as itr
from functools import wraps

import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

import matplotlib.pyplot as plt

from core import NullFunc, HARKobject, progress_timer
from interpolation import LinearInterp, BilinearInterp, ConstantFunction
from ConsIndShockModel import ConsIndShockSolver, constructAssetsGrid,\
                              IndShockConsumerType, KinkedRconsumerType
from simulation import drawLognormal
from utilities import CRRAutility, CRRAutilityP, CRRAutilityPP, \
                      CRRAutilityP_inv, CRRAutility_invP, \
                      CRRAutility_inv, CRRAutilityP_invP, plotFuncs 

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

class ConsIRASolution(HARKobject):
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
    
    def __init__(self, cFunc=None, dFunc=None, policyFunc=None, vFunc=None, 
                 vPfunc=None, vPPfunc=None, mNrmMin=None, hNrm=None, 
                 MPCmin=None, MPCmax=None):
        '''
        The constructor for a new ConsumerIRASolution object.

        Parameters
        ----------
        cFunc : function
            The consumption function for this period, defined over liquiud 
            market resources and illiquid account balance: c = cFunc(m,n).
        dFunc : function
            The optimal deposit/withdrawal function for this period, defined 
            over liquiud market resources and illiquid account balance: d = 
            dFunc(m,n).
        policyFunc : function
            Returns both the consumption and deposit functions in one
            calculation.
        vFunc : function
            The beginning-of-period value function for this period, defined 
            over liquiud market resources and illiquid account balance: 
            v = vFunc(m,n)
        vPfunc : function
            The beginning-of-period marginal value function, with respect to
            m, for this period, defined over liquiud market resources and 
            illiquid account balance: vP = vPfunc(m,n)
        vPPfunc : function
            The beginning-of-period marginal marginal value function, with 
            respect to m, for this period, defined over liquiud market 
            resources and illiquid account balance: vPP = vPPfunc(m,n)
        mNrmMin : float
            The minimum allowable liquid market resources for this period,
            conditional on having zero illiquid assets
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
        if policyFunc is None:
            policyFunc = NullFunc()
        if vFunc is None:
            vFunc = NullFunc()
        if vPfunc is None:
            vPfunc = NullFunc()
        if vPPfunc is None:
            vPPfunc = NullFunc()
        self.cFunc        = cFunc
        self.dFunc        = dFunc
        self.policyFunc   = policyFunc
        self.vFunc        = vFunc
        self.vPfunc       = vPfunc
        self.vPPfunc      = vPPfunc
        self.mNrmMin      = mNrmMin
        self.hNrm         = hNrm
        self.MPCmin       = MPCmin
        self.MPCmax       = MPCmax
        
class PureConsumptionFunc(HARKobject):
    '''
    A class for representing a pure consumption function. The underlying 
    interpolation is in the space of (l,b). If b is degenerate, uses
    LinearInterp. If b is not degenerate, uses BilinearInterp. When l <
    l_min(b), returns c = 0.
    '''
    distance_criteria = ['interpolator']

    def __init__(self,l_list,b_list,c_list,lMin,intercept_limit=None,
                 slope_limit=None):
        '''
        Constructor for a pure consumption function, c(l,b). Uses 1D
        interpolation when b is degenerate and 2D when b is not degenerate.

        Parameters
        ----------
        l_list : np.array
            (Normalized) grid of liquid market resource points for 
            interpolation.
        b_list : np.array
            (Normalized) grid of illiquid market resource points for 
            interpolation.
        c_list : np.array
            (Normalized) consumption points for interpolation.
        lMin : LinearInterp or ConstantFunction
            A function that returns the minimum level of l allowable, given b.
            For l < lMin(b), return c = 0.
        intercept_limit : float
            For linear interpolation. Intercept of limiting linear function.
        slope_limit : float
            For linear interpolation. Slope of limiting linear function.
            
        Returns
        -------
        None
        '''
        assert np.array(b_list >= 0).all(), 'b should be non-negative'
        self.bZero = np.array(b_list == 0.0).all()
        self.lMin  = deepcopy(lMin)
        
        if self.bZero: # b grid is degenerate
            self.interpolator = LinearInterp(l_list,c_list,intercept_limit,
                                             slope_limit)
        else: # b grid is not degenerate
            self.interpolator = BilinearInterp(c_list,l_list,b_list)

    def __call__(self,l,b=None):
        '''
        Evaluate the pure consumption function at given levels of liquid 
        market resources l and illiquid assets b. When b is degenerate at zero,
        function optionally takes one argument, l (liquid resources).

        Parameters
        ----------
        l : float or np.array
            Liquid market resources (normalized by permanent income).
        b : flot or np.array
            Illiquid market resources (normalized by permanent income)

        Returns
        -------
        c : float or np.array
            Pure consumption given liquid and illiquid market resources, 
            c(l,b).
        '''
        if np.array(b != None).all():
            assert np.array(b >= 0).all(), 'b should be non-negative'
        
        if self.bZero:
            c = self.interpolator(l)
        else:
            c = self.interpolator(l,b)
        
        # Set consumption to zero if l is below asset minimum
        c[l <= self.lMin(np.asarray(b))] = 0.0
        
        return c
    
class EndOfPeriodValueFunc(HARKobject):
    '''
    A class for representing the end-of-period value function, given end of 
    period assets a and b.  The underlying interpolation is in the space of 
    (a,b). If b is degenerate, uses LinearInterp. If b is not degenerate, uses 
    BilinearInterp.
    '''
    distance_criteria = ['interpolator']

    def __init__(self,a_list,b_list,w_list,aMin,uFunc,intercept_limit=None,
                 slope_limit=None):
        '''
        Constructor for a end-of-period value function, w(a,b).

        Parameters
        ----------
        a_list : np.array
            (Normalized) grid of liquid market resource points for 
            interpolation.
        b_list : np.array
            (Normalized) grid of illiquid market resource points for 
            interpolation.
        w_list : np.array
            Value points for interpolation.
        aMin : LinearInterp or ConstantFunction
            A function that returns the minimum level of a allowable given b.
            For a < aMin(b), return w = u(0.0001)
        uFunc: lambda function
            Flow utility function
        intercept_limit : float
            For linear interpolation. Intercept of limiting linear function.
        slope_limit : float
            For linear interpolation. Slope of limiting linear function.
            
        Returns
        -------
        None
        '''
        assert np.array(b_list >= 0).all(), 'b should be non-negative'
        self.bZero = np.array(b_list == 0.0).all()
        self.aMin = deepcopy(aMin)
        self.u = deepcopy(uFunc)
        
        if self.bZero: # b grid is degenerate
            self.interpolator = LinearInterp(a_list,w_list,intercept_limit,
                                             slope_limit)
        else: # b grid is not degenerate
            self.interpolator = BilinearInterp(w_list,a_list,b_list)

    def __call__(self,a,b):
        '''
        Evaluate the end-of-period value function at given levels of liquid 
        market resources a and illiquid assets b.

        Parameters
        ----------
        a : float or np.array
            Liquid market resources (normalized by permanent income).
        b : flot or np.array
            Illiquid market resources (normalized by permanent income)

        Returns
        -------
        w : float or np.array
            End-of-periord value given liquid and illiquid market resources, 
            w(a,b).
        '''
        assert np.array(b >= 0).all(), 'b should be non-negative'
        
        if self.bZero:
            w = self.interpolator(a)
        else:
            w = self.interpolator(a,b)
        
        # Set w to u(0.0001) if a is below asset minimum
        w[a <= self.aMin(np.asarray(b))] = self.u(0.0001)
            
        return w
            
class ConsIRAPolicyFunc(HARKobject):
    '''
    A class for representing the optimal consumption and deposit/withdrawal 
    functions.  The underlying interpolation is in the space of (m,n). If n is 
    degenerate, uses LinearInterp for consumption. If n is not degenerate, uses 
    BilinearInterp for consumption and deposit/withdrawal. Always obeys:
        
        l = m - (1-t(d))*d
        b = n + d
        c = c(l,b)
        
        t(d) = t*(d < 0)
        
    '''
    distance_criteria = ['m_list','n_list','d_list','cFuncPure']

    def __init__(self,m_list,n_list,d_list,MaxIRA,PenIRA,cFuncPure,
                 output='both'):
        '''
        Constructor for consumption and deposit/withdrawal functions, c(m,n)
        and d(m,n). Uses LinearInterp for c(m,n) interpolation when n is 
        degenerate and BilinearInterp when n is not degenerate.

        Parameters
        ----------
        m_list : np.array
            (Normalized) grid of liquid market resource points for 
            interpolation.
        n_list : np.array
            (Normalized) grid of illiquid market resource points for 
            interpolation.
        d_list : np.array
            (Normalized) deposit/withdrawal points for interpolation.
        MaxIRA : float
            (Nomralized) maximum allowable IRA deposit, d <= MaxIRA.
        PenIRA : float
            Penalty for withdrawing IRA in this period (could be zero)
        cFucnPure : float
            (Nomralized) consumption as a function of illiquid assets, l, and
            end-of-period illiquid assets, b.
            
        Returns
        -------
        None
        '''
        assert np.array(n_list >= 0).all(), 'n should be non-negative'
        self.nZero = np.array(n_list == 0.0).all()
        self.MaxIRA = MaxIRA
        self.PenIRA = PenIRA
        self.cFuncPure = deepcopy(cFuncPure)
        self.output = output
        
        self.m_list = m_list
        self.n_list = n_list
        self.d_list = d_list
        
        if not self.nZero: # n grid is not degenerate
            self.dInterpolator = BilinearInterp(d_list,m_list,n_list)

    def __call__(self,m,n):
        '''
        Evaluate the consumption and deposit/withdrawal function at given 
        levels of liquid market resources m and illiquid assets n.

        Parameters
        ----------
        m : float or np.array
            Liquid market resources (normalized by permanent income).
        n : flot or np.array
            Illiquid market resources (normalized by permanent income)

        Returns
        -------
        c : float or np.array
            Consumption given liquid and illiquid market resources, c(m,n).
        d : float or np.array
            Deposit/withdrawal given liquid and illiquid market resources, 
            d(m,n).
        '''
        if type(m) != np.ndarray:
            m = np.array([m])
        if type(n) != np.ndarray:
            n = np.array([n])
        assert np.array(n >= 0).all(), 'n should be non-negative'
        
        if self.nZero:
            c = self.cFuncPure(m,n)
            d = np.zeros(m.shape)
        else:
            d = self.dInterpolator(m,n)
            d[d < -n] = -n[d < -n]
            d[d > self.MaxIRA] = self.MaxIRA
            
            l = m - (1-self.PenIRA*(d < 0))*d
            b = n + d
            c = self.cFuncPure(l,b)
        
        if self.output == 'both':
            return c,d
        elif self.output == 'cFunc':
            return c
        elif self.output == 'dFunc':
            return d

class MultiValuedFunc(HARKobject):
    '''
    A class for representing a function f(x,y) = (g(x,y),h(x,y))
    '''
    distance_criteria = ['gFunc','hFunc']
    
    def __init__(self,gFunc,hFunc):
        '''
        Constructor for a multivalued function from domain (X,Y).
        
        Parameters
        ----------
        gFunc : function
            A real valued function with shared domain (X,Y).
        hFunc : function
            A real valued function with shared domain (X,Y).
        
        Returns
        -------
        none
        '''
        self.gFunc = deepcopy(gFunc)
        self.hFunc = deepcopy(hFunc)
    
    def __call__(self,x,y):
        '''
        Evaluate the g and h functions given x and y.
        
        Parameters
        ----------
        x : float or np.array
            First argument of g and h
        y : float or np.array
            Second argument of g and h
            
        Returns
        -------
        f : np.array of dimension 2
            f = (g(x,y),h(x,y))
        '''
        return self.gFunc(x,y), self.hFunc(x,y)
        

class ValueFuncIRA(HARKobject):
    '''
    A class for representing a value function.  The underlying interpolation is
    in the space of (m,n).
    '''
    distance_criteria = ['dfunc','PenIRA']

    def __init__(self,dFunc,makeNegvOfdFunc,PenIRA):
        '''
        Constructor for a new value function object.

        Parameters
        ----------
        dFunc : function
            A real function representing optimal deposit/withdrawal d given m 
            and n.
        makeNegvOfdFunc : function
           Calculate beginning-of-period value function given d, m, and n.
           Multiply it by -1 for use with minimizer tools, hence the "Neg".

        Returns
        -------
        None
        '''
        self.dFunc = deepcopy(dFunc)
        self.makeNegvOfdFunc = deepcopy(makeNegvOfdFunc)
        self.PenIRA = PenIRA

    def __call__(self,m,n):
        '''
        Evaluate the value function at given levels of liquid resources m and
        illiquid resource n. Since we use the "Neg" of the value function, we
        multiply it by -1 to get back the right-signed value function.

        Parameters
        ----------
        m : float or np.array
            Liquid market resources (normalized by permanent income).
        n : flot or np.array
            Illiquid market resources (normalized by permanent income)

        Returns
        -------
        v : float or np.array
            Lifetime value of beginning this period with liquid resources m and
            illiquid resources n; has same size as m, n.
        '''
        return -self.makeNegvOfdFunc(self.dFunc(m,n),m,n)

class MargValueFuncIRA(HARKobject):
    '''
    A class for representing a marginal value function in models where the
    standard envelope condition of dv(m,n)/dm = u'(c(m,n)) holds (with CRRA 
    utility)
    '''
    distance_criteria = ['cFunc','CRRA']

    def __init__(self,cFunc,CRRA):
        '''
        Constructor for a new marginal value function object.

        Parameters
        ----------
        cFunc : function
            A real function representing the consumption function.
        CRRA : float
            Coefficient of relative risk aversion.

        Returns
        -------
        None
        '''
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA

    def __call__(self,m,n):
        '''
        Evaluate the marginal value function at given levels of liquid 
        resources m and illiquid resources n.

        Parameters
        ----------
        m : float or np.array
            Liquid market resources (normalized by permanent income).
        n : flot or np.array
            Illiquid market resources (normalized by permanent income)

        Returns
        -------
        vP : float or np.array
            Marginal lifetime value of beginning this period with liquid 
            resources m and illiquid resources n; has same size as m, n.
        '''
        return utilityP(self.cFunc(m,n),gam=self.CRRA)

class TerminalValueFunc2D(HARKobject):
    '''
    A class for representing a terminal value function in a model with liquid
    assets and illiquid assets. The underlying interpolation is in the space of 
    (m,n) --> c.
    '''
    distance_criteria = ['cfunc','CRRA']

    def __init__(self,cFunc,CRRA):
        '''
        Constructor for a terminal value function object.

        Parameters
        ----------
        cFunc : function
            A real function representing the terminal consumption function
            defined on liquid market resources and illiquid market resources:
            c(m,n)
        CRRA : float
            Coefficient of relative risk aversion.

        Returns
        -------
        None
        '''
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA

    def __call__(self,m,n):
        '''
        Evaluate the value function at given levels of liquid market resources 
        m and illiquid assets n.

        Parameters
        ----------
        m : float or np.array
            Liquid market resources
        n : float or np.array
            Illiquid market resources

        Returns
        -------
        v : float or np.array
            Terminal value of beginning this period with liquid market 
            resources m and illiquid market resources n; has same size as 
            inputs m and p.
        '''
        return utility(self.cFunc(m,n),gam=self.CRRA)
        
class ConsIRAPFterminal(HARKobject):
    '''
    Closed form solution for IRA consumer with perfect foresight. Solution for 
    last period.
    '''
    distance_criteria = ['CRRA','PenIRA']
    
    def __init__(self,CRRA,PenIRA=0.0,output='all'):
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
        
        solution = {'cFunc': c, 'dFunc': d, 'aFunc': a, 'bFunc': b, 'vFunc': v, 
                    'vPmFunc': vPm, 'vPnFunc': vPn, 'max_state': max_state}
        
        if self.output == 'all':
            return solution
        else:
            return solution[self.output]

class ConsIRAPFnoPen(HARKobject):
    '''
    Solution for the IRA consumer with perfect foresight, during periods where
    there is no early withdrawal penalty.
    '''
    distance_criteria = ['period','NextIncome','DiscFac','CRRA','Rsave',
                         'Rira','MaxIRA','ConsIRAnext']
    
    def __init__(self,NextIncome,DiscFac,CRRA,Rsave,Rira,MaxIRA,
                 ConsIRAnext,output='all'):
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
        solLiq = self.ConsIRAnext(yN,np.zeros(n.shape))
        uPliq = uP(m + n)
        
        # lower bound on withdrawal is binding        
        liq = uPliq >= r*beta*solLiq['vPnFunc']
        
        if np.sum(liq) > 0: # if no one liquidates, skip
            c[...,0][liq] = m[liq] + n[liq]
            d[...,0][liq] = -n[liq]
            a[...,0][liq] = 0.0
            b[...,0][liq] = 0.0
            v[...,0][liq] = u(c[...,0][liq]) + beta*solLiq['vFunc'][liq]
            vPm[...,0][liq] = uP(c[...,0][liq])
            vPn[...,0][liq] = uP(c[...,0][liq])
            
        # Interior solution, partial illiquid withdrawal or deposit,
        # no liquid saving
        solCap = self.ConsIRAnext(yN,r*(n+dMax))
        # if m <= dMax, will not reach illiquid savings cap
        uPcap = np.where(m > dMax,uP(m - dMax),np.inf)
        
        inter = ((uPliq < r*beta*solLiq['vPnFunc']) &
                 (uPcap > r*beta*solCap['vPnFunc']))
        
        if np.sum(inter) > 0: # if no one is at interior solution, skip        
            # loop through solutions for values of m,n,yN and create an array
            d[...,1][inter] = np.array([br(self.dFOC,-ni,mi,args=(mi,ni,yNi))
                                        for mi,ni,yNi
                                        in itr.izip(m[inter].flatten(),
                                                    n[inter].flatten(),
                                                    yN[inter].flatten())    
                                        ]).reshape(m[inter].shape)
            
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
        cap = ((uPcap <= r*beta*solCap['vPnFunc']) & 
               (uPcap >= ra*beta*solCap['vPmFunc']) &
               (m > dMax))
        
        if np.sum(cap) > 0: # if no one is at cap w/ no liquid savings, skip               
            c[...,2][cap] = m[cap] - dMax
            d[...,2][cap] = dMax
            a[...,2][cap] = 0.0
            b[...,2][cap] = n[cap] + dMax
            v[...,2][cap] = u(c[...,2][cap]) + beta*solCap['vFunc'][cap]
            vPm[...,2][cap] = uP(c[...,2][cap])
            vPn[...,2][cap] = r*beta*solCap['vPnFunc'][cap]
        
        # Illiquid savings cap & liquid savings
        
        # upper bound on deposits binds and lower bound on liquid savings 
        # doesn't bind
        # cap on illiquid savings exceeds cash on hand
        cap_save = ((uPcap <= r*beta*solCap['vPnFunc']) & 
                    (uPcap < ra*beta*solCap['vPmFunc']) &
                    (m > dMax))
        
        if np.sum(cap_save) > 0: # if no one is at cap w/ liquid savings, skip
            # loop through solutions for values of m,n,yN and create an array
            a[...,3][cap_save] = np.array([br(self.aFOC,0.0,mi-dMax,
                                              args=(mi,ni,yNi))
                                           for mi,ni,yNi
                                           in itr.izip(m[cap_save].flatten(),
                                                       n[cap_save].flatten(),
                                                       yN[cap_save].flatten())
                                           ]).reshape(m[cap_save].shape)
                                           
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
        
class ConsIRAPFpen(HARKobject):
    '''
    Solution for the IRA consumer with perfect foresight, during periods where
    there is an early withdrawal penalty.
    '''
    distance_criteria = ['period','NextIncome','DiscFac','CRRA','Rsave',
                         'Rira','PenIRA','MaxIRA','ConsIRAnext']
    
    def __init__(self,NextIncome,DiscFac,CRRA,Rsave,Rira,PenIRA,MaxIRA,
                 ConsIRAnext,output='all'):
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
        solLiq = self.ConsIRAnext(yN,np.zeros(n.shape))
        uPliq = (1.0-t)*uP(m + (1.0-t)*n)

        # lower bound on withdrawal is binding        
        liq = uPliq >= r*beta*solLiq['vPnFunc']
        
        if np.sum(liq) > 0: # if no one liquidates, skip
            c[...,0][liq] = m[liq] + (1.0-t)*n[liq]
            d[...,0][liq] = -n[liq]
            a[...,0][liq] = 0.0
            b[...,0][liq] = 0.0
            v[...,0][liq] = u(c[...,0][liq]) + beta*solLiq['vFunc'][liq]
            vPm[...,0][liq] = uP(c[...,0][liq])
            vPn[...,0][liq] = (1.0-t)*uP(c[...,0][liq])       
            
        # Interior solution, partial illiquid withdrawal, no liquid saving
        solKink = self.ConsIRAnext(yN,r*n)
        uPwithdr = (1.0-t)*uP(m)
        
        # neither bound on withdrawals binds
        withdr = ((uPliq < r*beta*solLiq['vPnFunc']) & 
                  (uPwithdr > r*beta*solKink['vPnFunc']))
        
        # interior solution for withdrawal
        if np.sum(withdr) > 0: # if no one withdraws, skip
            # loop through solutions for values of m,n,yN and create an array
            d[...,1][withdr] = np.array([br(self.wFOC,-ni,0.0,
                                            args=(mi,ni,yNi)) 
                                        for mi,ni,yNi
                                        in itr.izip(m[withdr].flatten(),
                                                    n[withdr].flatten(),
                                                    yN[withdr].flatten())
                                         ]).reshape(m[withdr].shape)
        
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
        uPdep = uP(m)
        
        # upperbound on withdrawals and lower bound on deposits bind
        # lower bound on liquid savings binds
        kink = ((uPwithdr <= r*beta*solKink['vPnFunc']) &
                (uPdep >= r*beta*solKink['vPnFunc']) &
                (uPdep >= ra*beta*solKink['vPmFunc']))
        
        if np.sum(kink) > 0: # if no one is at kink, skip
            c[...,2][kink] = m[kink]
            d[...,2][kink] = 0.0
            a[...,2][kink] = 0.0
            b[...,2][kink] = n[kink]
            v[...,2][kink] = u(c[...,2][kink]) + beta*solKink['vFunc'][kink]
            vPm[...,2][kink] = uP(c[...,2][kink])
            vPn[...,2][kink] = r*beta*solKink['vPnFunc'][kink]
            
        # Corner solution w/ no illiquid withdrawal or saving & liquid saving
        
        # upperbound on withdrawals and lower bound on deposits bind
        # lower bound on liquid savings doesn't bind
        kink_save = ((uPwithdr <= r*beta*solKink['vPnFunc']) &
                     (uPdep >= r*beta*solKink['vPnFunc']) &
                     (uPdep < ra*beta*solKink['vPmFunc']))
        
        if np.sum(kink_save) > 0: # if no one saves at kink, skip
            # loop through solutions for values of m,n,yN and create an array
            a[...,3][kink_save] = np.array([br(self.aKinkFOC,0.0,mi,
                                               args=(mi,ni,yNi)) 
                                            for mi,ni,yNi
                                            in itr.izip(m[kink_save].flatten(),
                                                        n[kink_save].flatten(),
                                                        yN[kink_save].flatten()
                                                        )
                                            ]).reshape(m[kink_save].shape)        
        
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
        solCap = self.ConsIRAnext(yN,r*(n+dMax))
        
        # if m <= dMax, will not reach illiquid savings cap
        uPcap = np.where(m > dMax,uP(m - dMax),np.inf)
        
        # neither bound is binding for deposits
        dep = ((uPdep < r*beta*solKink['vPnFunc']) & 
               (uPcap > r*beta*solCap['vPnFunc']))
        
        if np.sum(dep) > 0: # if no one deposits, skip
            # loop through solutions for values of m,n,yN and create an array
            d[...,4][dep] = np.array([br(self.dFOC,0.0,mi,args=(mi,ni,yNi))
                                      for mi,ni,yNi
                                      in itr.izip(m[dep].flatten(),
                                                  n[dep].flatten(),
                                                  yN[dep].flatten())
                                      ]).reshape(m[dep].shape)
        
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
        cap = ((uPcap <= r*beta*solCap['vPnFunc']) & 
               (uPcap >= ra*beta*solCap['vPmFunc']) &
               (m > dMax))
        
        if np.sum(cap) > 0: # if no one is at cap w/ no liquid savings, skip
            c[...,5][cap] = m[cap] - dMax
            d[...,5][cap] = dMax
            a[...,5][cap] = 0.0
            b[...,5][cap] = n[cap] + dMax
            v[...,5][cap] = u(c[...,5][cap]) + beta*solCap['vFunc'][cap]
            vPm[...,5][cap] = uP(c[...,5][cap])
            vPn[...,5][cap] = r*beta*solCap['vPnFunc'][cap]
            
        # Illiquid savings cap and liquid savings
        
        # upper bound on deposits binds and lower bound on liquid savings 
        # doesn't bind
        # cap on illiquid savings exceeds cash on hand
        cap_save = ((uPcap <= r*beta*solCap['vPnFunc']) & 
                    (uPcap < ra*beta*solCap['vPmFunc']) &
                    (m > dMax))
        
        
        if np.sum(cap_save) > 0: # if no one is at cap w/ liquid savings, skip
            # loop through solutions for values of m,n,yN and create an array
            a[...,6][cap_save] = np.array([br(self.aFOC,0.0,mi - dMax,
                                              args=(mi,ni,yNi))
                                           for mi,ni,yNi
                                           in itr.izip(m[cap_save].flatten(),
                                                       n[cap_save].flatten(),
                                                       yN[cap_save].flatten())
                                           ]).reshape(m[cap_save].shape)
        
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
        
        solution = {'cFunc': c_star, 'dFunc': d_star, 'aFunc': a_star,
                    'bFunc': b_star, 'vFunc': v_star, 'vPmFunc': vPm_star,
                    'vPnFunc': vPn_star, 'max_state': max_state}
        
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
        
        # Corner solution with all assets placed in liquid account
        solLiq = self.ConsIRAnext(yN + ra*w,np.zeros(w.shape))
        
        liq = (ra*solLiq['vPmFunc'] > r*solLiq['vPnFunc'])
        
        if np.sum(liq) > 0: # if no one liquidates, skip
            c[...,0][liq] = 0.0
            d[...,0][liq] = 0.0
            a[...,0][liq] = w[liq]
            b[...,0][liq] = 0.0
            v[...,0][liq] = beta*solLiq['vFunc'][liq]
            vPm[...,0][liq] = ra*beta*solLiq['vPmFunc'][liq]
            vPn[...,0][liq] = 0.0
        
        # Interior solution with positive liquid and illiquid savings
        solCap = self.ConsIRAnext(yN,r*w)
        
        inter = ((ra*solLiq['vPmFunc'] <= r*solLiq['vPnFunc']) &
                 (ra*solCap['vPmFunc'] >= r*solCap['vPnFunc']))
        
        if np.sum(inter) > 0: # if no one is at interior, skip
            c[...,1][inter] = 0.0
            
            # loop through solutions for values of m,n,yN and create an array
            a[...,1][inter] = np.array([br(self.aFOC,0.0,wi,args=(wi,yNi))
                                        for wi,yNi
                                        in itr.izip(w[inter].flatten(),
                                                    yN[inter].flatten())    
                                        ]).reshape(w[inter].shape)

            d[...,1][inter] = w[inter] - a[...,1][inter]
            b[...,1][inter] = w[inter] - a[...,1][inter]
        
            solInter = self.ConsIRAnext(yN[inter] + ra*a[...,1][inter],
                                        r*(w[inter]-d[...,1][inter]))
        
            v[...,1][inter] = beta*solInter['vFunc']
            vPm[...,1][inter] = ra*beta*solInter['vPmFunc']
            vPn[...,1][inter] = 0.0
        
        # Corner solution with all asstes placed in illiquid account
        cap = ra*solCap['vPmFunc'] < r*solCap['vPnFunc']
        
        if np.sum(cap) > 0: # if no one is at cap, skip
            c[...,2][cap] = 0.0
            a[...,2][cap] = 0.0
            d[...,2][cap] = w[cap]
            b[...,2][cap] = w[cap]
            v[...,2][cap] = beta*solCap['vFunc'][cap]
            vPm[...,1][cap] = r*beta*solCap['vPnFunc'][cap]
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
        
        solution = {'cFunc': c_star, 'dFunc': d_star, 'aFunc': a_star,
                    'bFunc': b_star, 'vFunc': v_star, 'vPmFunc': vPm_star,
                    'vPnFunc': vPn_star, 'max_state': max_state}
        
        if self.output == 'all':
            return solution
        else:
            return solution[self.output]

# ====================================
# === Perfect foresight IRA model ===
# ====================================
        
class ConsIRAPFSolver(HARKobject):
    '''
    A class for solving a period of a perfect foresight consumption-saving 
    problem, with an illiquid and liquid account. 
    '''
    def __init__(self,solution_next,NextIncome,DiscFac,CRRA,Rsave,Rira,PenIRA,
                 MaxIRA,DistIRA,InitialProblem):
        '''
        Constructor for solver for prefect foresight consumption-saving
        problem with liquid and IRA-like illiquid savings account.
        
        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        NextIncome : float
            Income for the next period (known with certainty)
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
            Penalty for early withdrawals (d < 0) from the illiquid account, 
            i.e. before t = T_ira.
        MaxIRA: float
            Maximum allowable IRA deposit, d <= MaxIRA
        DistIRA: float or None
            Number of periods between current period and T_ira, i.e. T_ira - t.
            If DistIRA == None, T_ira > T_cycle, i.e. no expiration.
        InitialProblem: boolean
            If InitialProblem == 1, the first period involves an asset
            allocation decision, with no consumption decision or flow uitilty.
        
        Returns
        -------
        None
        '''
        assert Rira >= Rsave, 'Illiquid account must have higher return'
        
        # We ask that HARK users define single-letter variables they use in a 
        # dictionary attribute called notation.
        # Do that first.
        self.notation = {'a': 'liquid assets after all actions',
                         'b': 'illiquid assets after all actions',
                         'm': 'liquid market resources at decision time',
                         'n': 'illiduid market resources at decision time',
                         'c': 'consumption',
                         'd': 'illiquid deposit/withdrawal'}
        
        self.solution_next  = solution_next
        self.NextIncome     = NextIncome
        self.DiscFac        = DiscFac
        self.CRRA           = CRRA
        self.Rsave          = Rsave
        self.Rira           = Rira
        self.PenIRA         = PenIRA
        self.MaxIRA         = MaxIRA
        self.DistIRA        = DistIRA
        self.InitialProblem = InitialProblem
    
    def solve(self):
        '''
        Solves the one period, perfect foresight IRA consumption problem.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        solution: Function
            ConsIRAPFxxx solution function for this period.
        '''
        # If there is an initial allocation decision
        if self.InitialProblem:
            solution = ConsIRAPFinitial(self.NextIncome,self.DiscFac,self.CRRA,
                                        self.Rsave,self.Rira,
                                        self.solution_next)
        
        # During a period when there is still a penalty for withdrawals
        elif self.DistIRA < 0:
            solution = ConsIRAPFpen(self.NextIncome,self.DiscFac,self.CRRA,
                                    self.Rsave,self.Rira,self.PenIRA,
                                    self.MaxIRA,self.solution_next)
        
        # During a period when there is no penalty for withdrawals
        else:
            solution = ConsIRAPFnoPen(self.NextIncome,self.DiscFac,self.CRRA,
                                      self.Rsave,self.Rira,self.MaxIRA,
                                      self.solution_next)
        
        return solution
    
def solvePerfectForesightIRA(solution_next,NextIncome,DiscFac,CRRA,Rsave,Rira,
                             PenIRA,MaxIRA,DistIRA,InitialProblem):
    '''
    Solves a single period IRA consumer problem for a consumer with perfect
    foresight.
    
    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    NextIncome : float
        Income for the next period (known with certainty)
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
        Penalty for early withdrawals (d < 0) from the illiquid account, 
        i.e. before t = T_ira.
    MaxIRA: float
        Maximum allowable IRA deposit, d <= MaxIRA
    DistIRA: float or None
        Number of periods between current period and T_ira, i.e. T_ira - t.
        If DistIRA == None, T_ira > T_cycle, i.e. no expiration.
    InitialProblem: boolean
        If InitialProblem == 1, the first period involves an asset
        allocation decision, with no consumption decision or flow uitilty.
        
    Returns
    -------
    solution : ConsumerSolution
        The solution to this period's problem.
    '''
    solver = ConsIRAPFSolver(solution_next,NextIncome,DiscFac,CRRA,Rsave,Rira,
                             PenIRA,MaxIRA,DistIRA,InitialProblem)
    solution = solver.solve()
    return solution
        
# ==========================
# === General IRA model ===
# ==========================
        
class ConsIRASolver(ConsIndShockSolver):
    '''
    A class for solving a single period of a consumption-savings problem with
    a liquid savings account, and an IRA-like illiquid savings account. Model
    features constant relative risk aversion utility, permanent and transitory
    shocks to income, different interest rates for borowing and saving in the
    liquid account, and a separate interest rate for the illiquid acocunt.
    
    Inherits from ConsIndShockSolver, with additional inputs Rboro, Rsave, and
    Rira, which satsify the restriction that Rboro > Rira > Rsave. Also 
    requires the early withdrawal penalty, PenIRA, and a grid for illiquid 
    balance, bXtraGrid.
    '''
    def __init__(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rboro,Rsave,
                     Rira,PenIRA,MaxIRA,DistIRA,PermGroFac,BoroCnstArt,
                     aXtraGrid,bXtraGrid,lXtraGrid,vFuncBool,CubicBool,
                     ParallelBool):
        '''
        Constructor for a new solver for problems with risky income, a liquid
        and IRA-like illiquid savings account, different interest rates on 
        liquid borrowing/saving and illiquid saving.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn : [np.array]
            A list containing three arrays of floats, representing a discrete
            approximation to the income process between the period being solved
            and the one immediately following (in solution_next). Order: event
            probabilities, permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        DiscFac : float
            Intertemporal discount factor for future utility.
        CRRA : float
            Coefficient of relative risk aversion.
        Rboro: float
            Interest factor on liquid assets between this period and the 
            succeeding period when assets are negative.
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
        DistIRA: float or None
            Number of periods between current period and T_ira, i.e. T_ira - t.
            If DistIRA == None, T_ira > T_cycle, i.e. no expiration.
        PermGroFac : float
            Expected permanent income growth factor at the end of this period.
        BoroCnstArt: float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.  If it is less than the natural borrowing constraint,
            then it is irrelevant; BoroCnstArt=None indicates no artificial 
            borrowing constraint.
        aXtraGrid: np.array
            Array of "extra" end-of-period liquid asset values-- assets above 
            the absolute minimum acceptable level.
        bXtraGrid: np.array
            Array of "extra" end-of-period illiquid asset values-- assets above 
            the absolute minimum acceptable level.
        lXtraGrid: np.array
            Array of "extra" liquid assets just before the consumption decision
            -- assets above the abolute minimum acceptable level.
        vFuncBool: boolean
            An indicator for whether the value function should be computed and
            included in the reported solution.
        CubicBool: boolean
            An indicator for whether the solver should use cubic or linear 
            interpolation.
        ParallelBool: boolean
            An indicator for whether the solver should use parallel processing
            when solving for the optimal deposit amount over a grid of m and
            n values. Solver takes significantly longer otherwise.

        Returns
        -------
        None
        '''
        assert Rboro >= Rira >= Rsave, 'Interest factors must satisfy \
                                       Rboro>=Rira>=Rsave'
        assert vFuncBool == True, 'Must calculate value function \
                                      vFuncBool == True'
        
        # We ask that HARK users define single-letter variables they use in a 
        # dictionary attribute called notation.
        # Do that first.
        self.notation = {'a': 'liquid assets after all actions',
                         'b': 'illiquid assets after all actions',
                         'm': 'liquid market resources at decision time',
                         'n': 'illiduid market resources at decision time',
                         'l': 'liquid market resource at decision time, net \
                               of illiquid deposits/withdrawals',
                         'c': 'consumption',
                         'd': 'illiquid deposit/withdrawal'}
        
        # Initialize the solver.  Most of the steps are exactly the same as in
        # ConsIndShock case, so start with that.
        ConsIndShockSolver.__init__(self,solution_next,IncomeDstn,LivPrb,
                                   DiscFac,CRRA,Rboro,PermGroFac,BoroCnstArt,
                                   aXtraGrid,vFuncBool,CubicBool)
        
        # Assign factors, additional asset grids, IRA penalty, and time to IRA
        # penalty.
        self.Rboro        = Rboro
        self.Rsave        = Rsave
        self.Rira         = Rira
        self.PenIRA       = PenIRA
        self.MaxIRA       = MaxIRA
        self.DistIRA      = DistIRA
        self.bXtraGrid    = bXtraGrid
        self.lXtraGrid    = lXtraGrid
        self.ParallelBool = ParallelBool
        
    def defBoroCnst(self,BoroCnstArt):
        '''
        Calculates the borrowing constraint, conditional on the amount of
        normalized assets in the illiquid account. Uses the artificial and 
        natural borrowing constraints.

        Parameters
        ----------
        BoroCnstArt : float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.  If it is less than the natural borrowing constraint,
            then it is irrelevant; BoroCnstArt=None indicates no artificial 
            borrowing constraint.
            
        bXtraGrid : np.array
            Array of "extra" end-of-period illiquid asset values-- assets above 
            the absolute minimum acceptable level.

        Returns
        -------
        none
        '''
        if self.DistIRA == None: # There is no IRA penalty expiration
            bPDVFactor = (1.0 - self.PenIRA)*(self.Rira/self.Rboro)
            bPDVFactor_n = (1.0 - self.PenIRA)
        else:
            # Calculate PDV factor for illiquid assets next period when
            # a. account is liquidated next period
            if self.DistIRA > 1: # There is a penalty tomorrow
                bPDVFactorWithdrawNext = (1.0 - self.PenIRA)*(self.Rira/
                                                              self.Rboro)
            else: # No penalty tomorrow
                bPDVFactorWithdrawNext = (self.Rira/self.Rboro)
        
            # b. account isn't liquidated until T_ira
            if self.DistIRA > 0:
                bPDVFactorWithdrawT_ira = (self.Rira/self.Rboro)**self.DistIRA
            else:
                bPDVFactorWithdrawT_ira = (self.Rira/self.Rboro)
        
            # Calculate net value of illiquid assets liquidated at beginning of
            # period
            bPDVFactorWithdrawNow = (1.0 - self.PenIRA)
        
            # Take maximum PDV factor
            bPDVFactor = max(bPDVFactorWithdrawNext,bPDVFactorWithdrawT_ira)
            bPDVFactor_n = max(bPDVFactorWithdrawNow,bPDVFactorWithdrawT_ira)
        
        # Calculate the minimum allowable value of money resources in this 
        # period, when b = 0
        BoroCnstNat = ((self.solution_next.mNrmMin - self.TranShkMinNext)*
                           (self.PermGroFac*self.PermShkMinNext)/self.Rboro)
                           
        # Create natural borrowing constraint for different values of b
        self.BoroCnstNata = BoroCnstNat - np.append([0.0],
                                                    bPDVFactor*
                                                    np.asarray(self.bXtraGrid))
        
        self.BoroCnstNatn = BoroCnstNat - np.append([0.0],bPDVFactor_n*
                                                     np.asarray(self.bXtraGrid)
                                                     )
                           
        # Note: need to be sure to handle BoroCnstArt==None appropriately. 
        # In Py2, this would evaluate to 5.0:  np.max([None, 5.0]).
        # However in Py3, this raises a TypeError. Thus here we need to 
        # directly address the situation in which BoroCnstArt == None:
        if BoroCnstArt is None:
            self.mNrmMin = BoroCnstNat
            self.aNrmMinb = self.BoroCnstNata
            self.mNrmMinn = self.BoroCnstNatn
        else:
            self.mNrmMin = np.max([BoroCnstNat,BoroCnstArt])
            self.aNrmMinb = np.maximum(BoroCnstArt,self.BoroCnstNata)
            self.mNrmMinn = np.maximum(BoroCnstArt,self.BoroCnstNatn)
            
        if BoroCnstNat < self.mNrmMin: 
            self.MPCmaxEff = 1.0 # If actually constrained, MPC near limit is 1
        else:
            self.MPCmaxEff = self.MPCmaxNow
        
        # Calculates natural borrowing constraint
        if self.aNrmMinb.size == 1:
            self.BoroCnstFunc = ConstantFunction(self.aNrmMinb)
            self.BoroCnstFunc_n = ConstantFunction(self.mNrmMinn)
        else:
            self.BoroCnstFunc = LinearInterp(np.append([0],
                                             np.asarray(self.bXtraGrid)),
                                             self.aNrmMinb)
            self.BoroCnstFunc_n = LinearInterp(np.append([0],
                                             np.asarray(self.bXtraGrid)),
                                             self.mNrmMinn)
    
    def prepareToCalcEndOfPrdvAndvP(self):
        '''
        Prepare to calculate end-of-period value function and marginal value 
        by creating an array of liquid market resources and illiquid resources 
        that the agent could have next period, considering the grid of 
        end-of-period liquid and illiquid assets and the distribution of shocks 
        she might experience next period.

        Parameters
        ----------
        none

        Returns
        -------
        aNrmNow : np.array
            A 1D or 2D array of end-of-period liquid assets; also stored as 
            attribute of self.
        bNrmNow : np.array
            A 1D array of end-of-period illiquid assets; also stored as 
            attribute of self. Can potentially include only one element, when
            bXtraGrid = [].
        '''
        KinkBool = self.Rboro > self.Rsave # Boolean indicating that there is 
        # actually a kink. When Rboro == Rsave, this method reduces to a
        # standard consumption model. When Rboro < Rsave, the solver would have 
        # terminated when it was called.
        
        bNrmCount   = np.asarray(self.bXtraGrid).size + 1
        aNrmCount   = np.asarray(self.aXtraGrid).size
        bNrmNow     = np.insert(np.asarray(self.bXtraGrid),0,0.0)
        aNrmNow     = np.tile(np.asarray(self.aXtraGrid),(bNrmCount,1)) \
                        + np.transpose([self.aNrmMinb])
                 
        ShkCount    = self.TranShkValsNext.size
        aNrm_temp   = np.transpose(np.tile(aNrmNow,(ShkCount,1,1)),(1,0,2))
        bNrm_temp   = np.transpose(np.tile(bNrmNow[:,None],
                                           (aNrmCount,1,ShkCount)),(1,2,0))

        # Tile arrays of the income shocks and put them into useful shapes
        PermShkVals_temp  = np.transpose(np.tile(self.PermShkValsNext,
                                                 (bNrmCount,aNrmCount,1)),
                                                                    (0,2,1))
        TranShkVals_temp  = np.transpose(np.tile(self.TranShkValsNext,
                                                 (bNrmCount,aNrmCount,1)),
                                                                    (0,2,1))
        ShkPrbs_temp      = np.transpose(np.tile(self.ShkPrbsNext,
                                                 (bNrmCount,aNrmCount,1)),
                                                                    (0,2,1))
            
        # Make a 2D array of the interest factor at each asset gridpoint
        Rfree_Mat = self.Rsave*np.ones(aNrmNow.shape)
        if KinkBool:
            Rfree_Mat[aNrmNow < 0] = self.Rboro
            
        # Get liquid assets next period
        mNrmNext   = Rfree_Mat[:, None]/(self.PermGroFac*
                              PermShkVals_temp)*aNrm_temp + TranShkVals_temp
                            
        # Get illiquid assets nex period
        nNrmNext   = self.Rira/(self.PermGroFac*PermShkVals_temp)*bNrm_temp
        
        # If bXtragrid = [], remove unnecessary dimension from arrays
        if np.asarray(self.bXtraGrid).size == 0:
            aNrmNow           = aNrmNow[0]
            bNrmNow           = bNrmNow[0]
            mNrmNext          = mNrmNext[0]
            nNrmNext          = nNrmNext[0]
            PermShkVals_temp  = PermShkVals_temp[0]
            ShkPrbs_temp      = ShkPrbs_temp[0]
            TranShkVals_temp  = TranShkVals_temp[0]
            Rfree_Mat         = Rfree_Mat[0]
        
        # Recalculate the minimum MPC and human wealth using the interest 
        # factor on saving. This overwrites values from setAndUpdateValues, 
        # which were based on Rboro instead.
        if KinkBool:
            PatFacTop         = ((self.Rsave*self.DiscFacEff)**(1.0/self.CRRA)
                                                                   )/self.Rsave
            self.MPCminNow    = 1.0/(1.0 + PatFacTop/self.solution_next.MPCmin)
            self.hNrmNow      = self.PermGroFac/self.Rsave*(
                                                     np.dot(self.ShkPrbsNext,
                                                     self.TranShkValsNext
                                                     *self.PermShkValsNext) 
                                                     + self.solution_next.hNrm)

        # Store and report the results
        self.Rfree_Mat         = Rfree_Mat
        self.PermShkVals_temp  = PermShkVals_temp
        self.ShkPrbs_temp      = ShkPrbs_temp
        self.mNrmNext          = mNrmNext
        self.nNrmNext          = nNrmNext
        self.aNrmNow           = aNrmNow
        self.bNrmNow           = bNrmNow
        self.aNrmCount         = aNrmCount
        self.bNrmCount         = bNrmCount
        self.KinkBool          = KinkBool
        self.ShkCount          = ShkCount
        return aNrmNow, bNrmNow

    def calcEndOfPrdvAndvP(self,mNrmNext,nNrmNext,PermShkVals_temp,
                           ShkPrbs_temp,Rfree_Mat):
        '''
        Calculate end-of-period value function and marginal value function 
        for each point along the aNrmNow and bNrmNow grids. Does so by taking a 
        weighted sum of next period value function and marginal values across 
        income shocks (in a preconstructed grid self.mNrmNext and 
        self.nNrmNext).

        Parameters
        ----------
        none

        Returns
        -------
        EndOfPrdv  : np.array
            An array of value function levels, given end of period liquid and 
            illiquid assets.
        
        EndOfPrdvP : np.array
            An array of marginal value with respect to liquid assets, given 
            end of period liquid and illiquid assets.
        '''
        sum_axis = mNrmNext.ndim - 2
        
        Valid_m_n = mNrmNext >= self.BoroCnstFunc_n(nNrmNext)
        
        Censored_vFuncNext = np.where(Valid_m_n,self.vFuncNext(mNrmNext,
                                                               nNrmNext),
                                      self.u(0.0001))
        
        Censored_vPfuncNext = np.where(Valid_m_n,self.vPfuncNext(mNrmNext,
                                                                 nNrmNext),
                                       self.uP(0.0001))
        
        EndOfPrdv   = self.DiscFacEff*\
                            np.sum(PermShkVals_temp**
                                   (1.0-self.CRRA)*self.PermGroFac**
                                   (1.0-self.CRRA)*
                                   Censored_vFuncNext*
                                   ShkPrbs_temp,axis=sum_axis)
        
        EndOfPrdvP  = self.DiscFacEff*\
                            Rfree_Mat*\
                            self.PermGroFac**(-self.CRRA)*\
                            np.sum(PermShkVals_temp**(-self.CRRA)*\
                                   Censored_vPfuncNext*
                                   ShkPrbs_temp,axis=sum_axis)
        return EndOfPrdv, EndOfPrdvP

    def getPointsForPureConsumptionInterpolation(self,EndOfPrdv,EndOfPrdvP,
                                                 aNrmNow):
        '''
        Finds interpolation points (c,l,b) for the pure consumption function.
        Uses an upper envelope algorithm (Druedahl, 2018) to address potential
        nonconvexities.
        
        Parameters
        ----------
        EndOfPrdv : np.array
            Array of end-of-period value function levels.
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aNrmNow : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.
        lXtraGrid : np.array
            Array of "extra" liquid assets just before the consumption decision
            -- assets above the abolute minimum acceptable level. 

        Returns
        -------
        c_for_interpolation : np.array
            Consumption points for interpolation.
        l_for_interpolation : np.array
            Corresponding liquid market resource points for interpolation.
        '''
        cNrm_ik = self.uPinv(EndOfPrdvP)
        lNrm_ik = cNrm_ik + aNrmNow
        
        # Construct b-specific grids for l, including borrowing constraint
        # Then construct one grid for l, using non-overlapping segments of 
        # b-specific grids
        lNrm_jk = np.tile(np.insert(np.asarray(self.lXtraGrid),0,0.0),
                          (self.bNrmCount,1)) + np.transpose([self.aNrmMinb])
        
        if self.bNrmCount == 1:
            lNrm_j = lNrm_jk[0]
        else:
            lNrm_jk_Xtra = [lNrm_jk[i][lNrm_jk[i] < np.min(lNrm_jk[i-1])] 
                            for i in range(1,len(lNrm_jk))]
            lNrm_j = np.sort(np.append(lNrm_jk[0],np.hstack(lNrm_jk_Xtra)))
        
        lNrmCount = lNrm_j.size
        
        # Construct b_k x l_j specific grids for l,c,a, and w
        lNrm_ik_temp,cNrm_ik_temp,aNrm_ik_temp,w_ik_temp = \
            [np.transpose(np.tile(x,(lNrmCount,1,1)),(1,0,2)) for x in 
             [lNrm_ik,cNrm_ik,aNrmNow,EndOfPrdv]]
        
        # Find where l_j is in [l_ik , l_i+1k]
        lNrm_j_temp = np.tile(lNrm_j[:,None],(self.bNrmCount,1,1))
        lNrm_j_mask = (lNrm_j_temp > lNrm_ik_temp[:,:,:-1]) \
                        & ~(lNrm_j_temp > lNrm_ik_temp[:,:,1:])
        
        
        i = [[np.flatnonzero(row) for row in mat] for mat in lNrm_j_mask]
        
        # Calculate candidate optimal consumption, c_j_ik
        # Calculate associated assets, a_j_ik, and next period value, w_j_ik
        # Find consumption that maximizes utility
        cNrm_j_ik = [[c[t] + (c[t+1] - c[t])/(l[t+1] - l[t])*(lj - l[t])
                     if t.size > 0 else np.array([]) for c,l,lj,t in 
                     zip(ci,li,lji,ti)] for ci,li,lji,ti in zip(cNrm_ik_temp,
                                                                lNrm_ik_temp,
                                                                lNrm_j_temp,i)]
        
        aNrm_j_ik = [[l - c if c.size > 0 else np.array([]) for c,l in 
                     zip(ci,li)] for ci,li in zip(cNrm_j_ik,lNrm_j_temp)]
        
        w_j_ik = [[w[t] + (w[t+1] - w[t])/(a[t+1] - a[t])*(aj - a[t])
                  if t.size > 0 else np.array([]) for w,a,aj,t in
                  zip(wi,ai,aji,ti)] for wi,ai,aji,ti in zip(w_ik_temp,
                                                             aNrm_ik_temp,
                                                             aNrm_j_ik,i)]
        
        v_j_ik = self.u(np.asarray(cNrm_j_ik)) + np.asarray(w_j_ik)
        
        cNrm_j_k = [[c[np.argmax(v)] if c.size > 0 else 0.0 for c,v in 
                    zip(ci,vi)] for ci,vi in zip(cNrm_j_ik,v_j_ik)]
        
        if self.bNrmCount == 1:
            c_for_interpolation = np.array(cNrm_j_k)[0]
        else:
            c_for_interpolation = np.transpose(np.array(cNrm_j_k))
        
        l_for_interpolation = lNrm_j
        
        return c_for_interpolation, l_for_interpolation
    
    def makePurecFunc(self,cNrm,lNrm,bNrm):
        '''
        Constructs a pure consumption function c(l,b), i.e. optimal consumption
        given l, holding b fixed this period (no deposits or withdrawals), to
        be used by other methods.

        Parameters
        ----------
        cNrm : np.array
            (Normalized) consumption points for interpolation.
        lNrm : np.array
            (Normalized) grid of liquid market resource points for 
            interpolation.
        bNrm : np.array
            (Normalized) grid of illiquid market resource points for 
            interpolation.

        Returns
        -------
        none
        '''
        cFuncNowPure = PureConsumptionFunc(lNrm,bNrm,cNrm,self.BoroCnstFunc,
                                           self.MPCminNow*self.hNrmNow,
                                           self.MPCminNow)
        self.cFuncNowPure = cFuncNowPure
    
    def makeEndOfPrdvFunc(self,EndOfPrdv):
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
        if self.bNrmCount == 1:
            self.EndOfPrdvFunc = EndOfPeriodValueFunc(self.aNrmNow,
                                                      self.bNrmNow,EndOfPrdv,
                                                      self.BoroCnstFunc,self.u)
            self.aNrmNowUniform = self.aNrmNow
        else:
            aNrmNow_Xtra = [self.aNrmNow[i][self.aNrmNow[i] < 
                                     np.min(self.aNrmNow[i-1])] for i in
                                range(1,len(self.aNrmNow))]
            aNrmNowUniform = np.sort(np.append(self.aNrmNow[0],
                                             np.hstack(aNrmNow_Xtra)))
            aNrm = np.tile(aNrmNowUniform,(self.bNrmCount,1))
            aNrmCount = aNrmNowUniform.size
        
        
            aNrm_temp = np.transpose(np.tile(aNrm,(self.ShkCount,1,1)),(1,0,2))
            bNrm_temp = np.transpose(np.tile(self.bNrmNow[:,None],
                                             (aNrmCount,1,
                                              self.ShkCount)),(1,2,0))
        
            # Tile arrays of the income shocks and put them into useful shapes
            PermShkVals_temp  = np.transpose(np.tile(self.PermShkValsNext,
                                                 (self.bNrmCount,aNrmCount,1)),
                                                                    (0,2,1))
            TranShkVals_temp  = np.transpose(np.tile(self.TranShkValsNext,
                                                 (self.bNrmCount,aNrmCount,1)),
                                                                    (0,2,1))
            ShkPrbs_temp      = np.transpose(np.tile(self.ShkPrbsNext,
                                                 (self.bNrmCount,aNrmCount,1)),
                                                                    (0,2,1))
        
            # Make a 2D array of the interest factor at each asset gridpoint
            Rfree_Mat = self.Rsave*np.ones(aNrm.shape)
            if self.KinkBool:
                Rfree_Mat[aNrm < 0] = self.Rboro
        
            # Get liquid assets next period
            mNrmNext   = Rfree_Mat[:, None]/(self.PermGroFac*
                              PermShkVals_temp)*aNrm_temp + TranShkVals_temp
                            
            # Get illiquid assets nex period
            nNrmNext   = self.Rira/(self.PermGroFac*PermShkVals_temp)*bNrm_temp
        
            # Calculate end of period value functions
            EndOfPrdvUniform, _ = self.calcEndOfPrdvAndvP(mNrmNext,nNrmNext,
                                               PermShkVals_temp,ShkPrbs_temp,
                                               Rfree_Mat)
        
            EndOfPrdv_trans = np.transpose(EndOfPrdvUniform)
        
            self.EndOfPrdvFunc = EndOfPeriodValueFunc(aNrmNowUniform,
                                                      self.bNrmNow,
                                                  EndOfPrdv_trans,
                                                  self.BoroCnstFunc,self.u)
            self.aNrmNowUniform = aNrmNowUniform
        
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
        cNrm = self.cFuncNowPure(lNrm,bNrm)
        aNrm = lNrm - cNrm
        
        # can't actually evaluate cNrm == 0
        c_pos = np.where(np.asarray(cNrm)>0.0,cNrm,0.0001)
        
        v = self.u(c_pos) + self.EndOfPrdvFunc(aNrm,bNrm)
        
        return -v
    
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
        d = basinhopping(self.makeNegvOfdFunc,
                         0.0,minimizer_kwargs={"bounds":((-nNrm + 1e-10,
                                                          self.MaxIRA),),
                                               "args":(mNrm,nNrm)}).x
        
        return d
        
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
        if self.bNrmCount == 1:
            self.dFuncNow = ConstantFunction(0.0)
            self.cFuncNow = self.cFuncNowPure
            self.policyFuncNow = MultiValuedFunc(self.cFuncNowPure,
                                                 ConstantFunction(0.0))
        else:
            mNrm = self.aNrmNowUniform
            nNrm = self.bNrmNow
            
            n_repeat = np.repeat(np.array(nNrm),len(mNrm))
            m_tile = np.tile(np.array(mNrm),len(nNrm))
           
            if self.ParallelBool:
                n_cpus = mp.cpu_count()
                pool = ProcessPool(processes=max(n_cpus-1,1))
                dNrm_list = pool.map(self.findArgMaxv, m_tile, n_repeat)
            else:
                dNrm_list = [[self.findArgMaxv(m,n) for m in mNrm] 
                                                    for n in nNrm]
            
            dNrm = np.array(dNrm_list).reshape(len(nNrm),len(mNrm))
            dNrm_trans = np.transpose(dNrm)
            
            self.cFuncNow = ConsIRAPolicyFunc(mNrm,nNrm,dNrm_trans,self.MaxIRA,
                                              self.PenIRA,self.cFuncNowPure,
                                              output='cFunc')
            self.dFuncNow = ConsIRAPolicyFunc(mNrm,nNrm,dNrm_trans,self.MaxIRA,
                                              self.PenIRA,self.cFuncNowPure,
                                              output='dFunc')
            self.policyFuncNow = ConsIRAPolicyFunc(mNrm,nNrm,dNrm_trans,
                                                  self.MaxIRA,self.PenIRA,
                                                  self.cFuncNowPure,
                                                  output='both')
    
    def makeBasicSolution(self,EndOfPrdv,EndOfPrdvP,aNrm,bNrm):
        '''
        Given end of period assets and end of period marginal value, construct
        the basic solution for this period.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aNrm : np.array
            Array of end-of-period liquid asset values.
        bNrm : np.array
            Array of end-of-period illiquid asset values.

        Returns
        -------
        solution_now : ConsIRASolution
            The solution to this period's consumption-saving problem, with a
            consumption function, deposit function, value fucntion, marginal 
            value function, and minimum m.
        '''
        cNrm,lNrm = self.getPointsForPureConsumptionInterpolation(EndOfPrdv,
                                                                  EndOfPrdvP,
                                                                  aNrm)
        self.makePurecFunc(cNrm,lNrm,bNrm)
        self.makeEndOfPrdvFunc(EndOfPrdv)
        self.makePolicyFunc()
        vFuncNow = ValueFuncIRA(self.dFuncNow,self.makeNegvOfdFunc,self.PenIRA)
        vPfuncNow = MargValueFuncIRA(self.cFuncNow,self.CRRA)
        solution_now = ConsIRASolution(cFunc = self.cFuncNow,
                                       dFunc = self.dFuncNow, 
                                       policyFunc = self.policyFuncNow,
                                       vFunc = vFuncNow,
                                       vPfunc = vPfuncNow, 
                                       mNrmMin = self.mNrmMin)
        return solution_now
    
    def addMPCandHumanWealth(self,solution):
        '''
        Take a solution and add human wealth and the bounding MPCs to it.

        Parameters
        ----------
        solution : ConsIRASolution
            The solution to this period's consumption-saving problem.

        Returns:
        ----------
        solution : ConsIRASolution
            The solution to this period's consumption-saving problem, but now
            with human wealth and the bounding MPCs.
        '''
        solution.hNrm   = self.hNrmNow
        solution.MPCmin = self.MPCminNow
        solution.MPCmax = self.MPCmaxEff
        return solution
        
    def solve(self):
        '''
        Solves a one period consumption saving problem with liquid and illiquid
        assets.

        Parameters
        ----------
        None

        Returns
        -------
        solution : ConsIRASolution
            The solution to the one period problem.
        '''
        aNrm,bNrm = self.prepareToCalcEndOfPrdvAndvP()
        EndOfPrdv,EndOfPrdvP = self.calcEndOfPrdvAndvP(self.mNrmNext,
                                                        self.nNrmNext,
                                                        self.PermShkVals_temp,
                                                        self.ShkPrbs_temp,
                                                        self.Rfree_Mat)
        solution = self.makeBasicSolution(EndOfPrdv,EndOfPrdvP,aNrm,bNrm)
        solution = self.addMPCandHumanWealth(solution)
        return solution

def solveConsIRA(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rboro,Rsave,Rira,
                 PenIRA,MaxIRA,DistIRA,PermGroFac,BoroCnstArt,aXtraGrid,
                 bXtraGrid,lXtraGrid,vFuncBool,CubicBool,ParallelBool):
    '''
    Solves a single period consumption-saving problem with CRRA utility and 
    risky income (subject to permanent and transitory shocks), with liquid and
    illiquid assets.

    Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn : [np.array]
            A list containing three arrays of floats, representing a discrete
            approximation to the income process between the period being solved
            and the one immediately following (in solution_next). Order: event
            probabilities, permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        DiscFac : float
            Intertemporal discount factor for future utility.
        CRRA : float
            Coefficient of relative risk aversion.
        Rboro: float
            Interest factor on liquid assets between this period and the 
            succeeding period when assets are negative.
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
        DistIRA: float or None
            Number of periods between current period and T_ira, i.e. T_ira - t
        PermGroFac : float
            Expected permanent income growth factor at the end of this period.
        BoroCnstArt: float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.  If it is less than the natural borrowing constraint,
            then it is irrelevant; BoroCnstArt=None indicates no artificial 
            borrowing constraint.
        aXtraGrid: np.array
            Array of "extra" end-of-period liquid asset values-- assets above 
            the absolute minimum acceptable level.
        bXtraGrid: np.array
            Array of "extra" end-of-period illiquid asset values-- assets above 
            the absolute minimum acceptable level.
        lXtraGrid: np.array
            Array of "extra" liquid assets just before the consumption decision
            -- assets above the abolute minimum acceptable level.
        vFuncBool: boolean
            An indicator for whether the value function should be computed and
            included in the reported solution.
        CubicBool: boolean
            An indicator for whether the solver should use cubic or linear 
            interpolation.
        ParallelBool: boolean
            An indicator for whether the solver should use parallel processing
            when solving for the optimal deposit amount over a grid of m and
            n values. Solver takes significantly longer otherwise.

    Returns
    -------
    solution_now : ConsIRASolution
        The solution to the single period consumption-saving problem with
        liquid and illiquid assets.  Includes a consumption function cFunc, 
        deposit function dFunc, a value function vFunc, a marginal, value 
        function vPfunc, a minimum acceptable level of liquid resources given 
        zero illiquid resources mNrmMin, normalized human wealth hNrm, and 
        bounding MPCs MPCmin and MPCmax.
    '''
    solver = ConsIRASolver(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rboro,
                           Rsave,Rira,PenIRA,MaxIRA,DistIRA,PermGroFac,
                           BoroCnstArt,aXtraGrid,bXtraGrid,lXtraGrid,vFuncBool,
                           CubicBool,ParallelBool)
    solver.prepareToSolve()       # Do some preparatory work
    solution_now = solver.solve() # Solve the period
    return solution_now

class IRAConsumerType(IndShockConsumerType):
    '''
    A consumer type that faces idiosyncratic shocks to income and has a liquid
    and illiquid savings account, with different interest factors on saving vs 
    borrowing in the liquid account, and a different interest factor on the
    illiquid account. Extends IndShockConsumerType, but uses a different
    solution concept, the Nested Endogenous Grid Method (NEGM). Solver for this 
    class is currently only compatible with linear spline interpolation.
    '''
    cFunc_terminal_ = BilinearInterp(np.array([[0.0,1.0],[1.0,2.0]]),
                                     np.array([0.0,1.0]),np.array([0.0,1.0]))
    dFunc_terminal_ = BilinearInterp(np.array([[0.0,-1.0],[0.0,-1.0]]),
                                     np.array([0.0,1.0]),np.array([0.0,1.0]))
    policyFunc_terminal_ = MultiValuedFunc(cFunc_terminal_,dFunc_terminal_)
    solution_terminal = ConsIRASolution(cFunc = cFunc_terminal_,
                                        dFunc = dFunc_terminal_,
                                        policyFunc = policyFunc_terminal_,
                                        mNrmMin=0.0,hNrm=0.0,MPCmin=1,MPCmax=1)
    
    time_inv_ = copy(IndShockConsumerType.time_inv_)
    time_inv_.remove('Rfree')
    time_inv_ += ['Rboro', 'Rsave','Rira','MaxIRA','ParallelBool']
    
    time_vary_ = IndShockConsumerType.time_vary_
    
    poststate_vars_ = ['aNrmNow','bNrmNow','pLvlNow']
    

    def __init__(self,cycles=1,time_flow=True,**kwds):
        '''
        Instantiate a new ConsumerType with given data. See 
        ConsumerParameters.init_IRA for a dictionary of the keywords that 
        should be passed to the constructor.

        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.
        time_flow : boolean
            Whether time is currently "flowing" forward for this instance.

        Returns
        -------
        None
        '''
        # Initialize a basic AgentType
        IndShockConsumerType.__init__(self,cycles=cycles,time_flow=time_flow
                                      ,**kwds)

        # Add consumer-type specific objects, copying to create independent 
        # versions
        self.solveOnePeriod = solveConsIRA # IRA solver
        self.update() # Make assets grid, income process, terminal solution,
                      # PenIRA, DistIRA, and Parallel status
                      
    def updateSolutionTerminal(self):
        '''
        Update the terminal period solution.  This method should be run when a
        new AgentType is created or when CRRA changes.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        if self.T_ira < self.T_cycle:
            cFunc_terminal_ = BilinearInterp(np.array([[0.0,1.0],[1.0,2.0]]),
                                             np.array([0.0,1.0]),
                                             np.array([0.0,1.0]))
        else:
            cFunc_terminal_ = BilinearInterp(np.array([[0.0,1.0 - 
                                                        self.PenIRAFixed],
                                                        [1.0,2.0 - 
                                                         self.PenIRAFixed]]),
                                             np.array([0.0,1.0]),
                                             np.array([0.0,1.0]))
        self.solution_terminal.cFunc = cFunc_terminal_
        self.solution_terminal.policyFunc = MultiValuedFunc(cFunc_terminal_,
                                                           self.dFunc_terminal_
                                                           )
        self.solution_terminal.vFunc   = TerminalValueFunc2D(cFunc_terminal_,
                                                             self.CRRA)
        self.solution_terminal.vPfunc  = MargValueFuncIRA(cFunc_terminal_,
                                                          self.CRRA)
        
    def update(self):
        '''
        Update the income process, the assets grids, the IRA penalty, distance
        to IRA penalty expiration, and the terminal solution.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        IndShockConsumerType.update(self)
        self.updatelGrid()
        self.updatebGrid()
        self.updateIRA()
        
    def updatelGrid(self):
        '''
        Update the grid for l, assets net of deposits/withdrawals.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.lXtraGrid = self.aXtraGrid
        self.addToTimeInv('lXtraGrid')
        
    def updatebGrid(self):
        '''
        Update the grid for b, illiquid assets.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        if self.bXtraCount == 0:
            self.bXtraGrid = np.array([])
        else:
            bgrid = HARKobject()
            bgrid.aXtraMin = self.bXtraMin
            bgrid.aXtraMax = self.bXtraMax
            bgrid.aXtraCount = self.bXtraCount
            bgrid.aXtraNestFac = self.bXtraNestFac
            bgrid.aXtraExtra = np.array([None])
            self.bXtraGrid = constructAssetsGrid(bgrid)
        self.addToTimeInv('bXtraGrid')
        
    def updateIRA(self):
        '''
        Create the time pattern of IRA penalties and distance from IRA
        expiration.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        if self.T_ira < 0: # No penalty
            self.PenIRA = self.T_cycle*[0.0]
            self.DistIRA = [self.T_ira - t for t in range(self.T_cycle)]
        elif self.T_ira < self.T_cycle: # Penalty for part of the time
            self.PenIRA = (self.T_ira)*[self.PenIRAFixed] +\
                          (self.T_cycle - self.T_ira)*[0.0]
            self.DistIRA = [self.T_ira - t for t in range(self.T_cycle)]
        else: # Penaltly all the time
            self.PenIRA = self.T_cycle*[self.PenIRAFixed]
            self.DistrIRA = self.T_cycle*[None]
        self.addToTimeVary('PenIRA','DistIRA')
                                   
    def getRfree(self):
        '''
        Returns an array of size self.AgentCount with self.Rboro or self.Rsave 
        in each entry, based on whether self.aNrmNow >< 0.

        Parameters
        ----------
        None

        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for 
             each agent.
        '''
        RfreeNow = self.Rboro*np.ones(self.AgentCount)
        RfreeNow[self.aNrmNow > 0] = self.Rsave
        return RfreeNow
    
    def getRill(self):
        '''
        Returns an array of size self.AgentCount with self.Rira in each entry.
        
        Parameters
        ----------
        None

        Returns
        -------
        RillNow : np.array
             Array of size self.AgentCount with illiquid asset return factor
             for each agent.
        '''
        RillNow = self.Rira*np.ones(self.AgentCount)
        return RillNow
        
    def getStates(self):
        '''
        Calculates updated values of normalized liquid and illiquid market 
        resources and permanent income level for each agent.  Uses pLvlNow, 
        aNrmNow, bNrmNow, PermShkNow, TranShkNow.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        pLvlPrev = self.pLvlNow
        aNrmPrev = self.aNrmNow
        bNrmPrev = self.bNrmNow
        RfreeNow = self.getRfree()
        RillNow = self.getRill()

        # Calculate new states: normalized liquid and illiquid market 
        # resources and permanent income level
        self.pLvlNow = pLvlPrev*self.PermShkNow # Updated permanent income 
                                                # level
        self.PlvlAggNow = self.PlvlAggNow*self.PermShkAggNow # Updated 
                                                             # aggregate 
                                                             # permanent 
                                                             # productivity 
                                                             # level
        ReffNow = RfreeNow/self.PermShkNow # "Effective" interest factor on 
                                           # normalized liquid assets
        self.mNrmNow = ReffNow*aNrmPrev + self.TranShkNow # Liquid Market 
                                                          # resources after 
                                                          # income
        ReffIllNow = RillNow/self.PermShkNow # "Effective" interest factor on 
                                             # normalized illiquid assets
        self.nNrmNow = ReffIllNow*bNrmPrev
        
    def getControls(self):
        '''
        Calculates consumption and deposit/withdrawal for each consumer of this 
        type using the consumption and deposit/withdrawal functions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        dNrmNow = np.zeros(self.AgentCount) + np.nan
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cNrmNow[these], dNrmNow[these] = self.solution[t].policyFunc(
                                                           self.mNrmNow[these],
                                                           self.nNrmNow[these])
        self.cNrmNow = cNrmNow
        self.cLvlNow = cNrmNow*self.pLvlNow
        self.dNrmNow = dNrmNow
        self.dLvlNow = dNrmNow*self.pLvlNow
        return None
    
    def getPostStates(self):
        '''
        Calculates end-of-period liquid and illiquid assets for each consumer 
        of this type.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        lNrmNow = np.zeros(self.AgentCount) + np.nan
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            lNrmNow[these] = self.mNrmNow[these] - \
                                  (1 - self.PenIRA[t]*\
                                  (self.dNrmNow[these] < 0))*\
                                  self.dNrmNow[these]
        
        self.aNrmNow = lNrmNow - self.cNrmNow
        self.aLvlNow = self.aNrmNow*self.pLvlNow   # Useful in some cases to 
                                                   # precalculate asset level
        
        self.bNrmNow = self.nNrmNow + self.dNrmNow
        self.bLvlNow = self.bNrmNow*self.pLvlNow
        return None
    
    def unpackdFunc(self):
        '''
        "Unpacks" the deposit/withdrawal functions into their own field for 
        easier access. After the model has been solved, the deposit functions 
        reside in the attribute dFunc of each element of ConsumerType.solution.  
        This method creates a (time varying) attribute dFunc that contains a 
        list of deposit functions.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        self.dFunc = []
        for solution_t in self.solution:
            self.dFunc.append(solution_t.dFunc)
        self.addToTimeVary('dFunc')
        
    def simBirth(self,which_agents):
        '''
        Makes new consumers for the given indices. Initialized variables 
        include aNrm, bNrm, and pLvl, as well as time variables t_age and 
        t_cycle.  Normalized assets and persistent income levels are drawn from 
        lognormal distributions given by aNrmInitMean and aNrmInitStd (etc).

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents 
            should be "born".

        Returns
        -------
        None
        '''
        # Get and store states for newly born agents
        N = np.sum(which_agents) # Number of new consumers to make
        self.aNrmNow[which_agents] = drawLognormal(N,mu=self.aNrmInitMean,
                                                   sigma=self.aNrmInitStd,
                                                   seed=
                                                   self.RNG.randint(0,2**31-1))
        # Account for newer cohorts having higher permanent income
        pLvlInitMeanNow = self.pLvlInitMean + np.log(self.PlvlAggNow) 
        self.pLvlNow[which_agents] = drawLognormal(N,mu=pLvlInitMeanNow,
                                                   sigma=self.pLvlInitStd,
                                                   seed=
                                                   self.RNG.randint(0,2**31-1))
        self.bNrmNow[which_agents] = 0.0
        self.t_age[which_agents]   = 0 # How many periods since each agent was 
                                       # born
        self.t_cycle[which_agents] = 0 # Which period of the cycle each agent 
                                       # is currently in
        
    def makeEulerErrorFunc(self,mMax=100,approx_inc_dstn=True):
        '''
        Creates a "normalized Euler error" function for this instance, mapping
        from market resources to "consumption error per dollar of consumption."
        Stores result in attribute eulerErrorFunc as an interpolated function.
        Has option to use approximate income distribution stored in 
        self.IncomeDstn or to use a (temporary) very dense approximation.

        NOT YET IMPLEMENTED FOR THIS CLASS

        Parameters
        ----------
        mMax : float
            Maximum normalized market resources for the Euler error function.
        approx_inc_dstn : Boolean
            Indicator for whether to use the approximate discrete income 
            distribution stored in self.IncomeDstn[0], or to use a very 
            accurate discrete approximation instead.  When True, uses 
            approximation in IncomeDstn; when False, makes and uses a very 
            dense approximation.

        Returns
        -------
        None
        '''
        raise NotImplementedError()

    def checkConditions(self,verbose=False):
        '''
        This method checks whether the instance's type satisfies the growth 
        impatiance condition (GIC), return impatiance condition (RIC), absolute 
        impatiance condition (AIC), weak return impatiance condition (WRIC), 
        finite human wealth condition (FHWC) and finite value of autarky 
        condition (FVAC). These are the conditions that are sufficient for 
        nondegenerate solutions under infinite horizon with a 1 period cycle. 
        Depending on the model at hand, a different combination of these 
        conditions must be satisfied. To check which conditions are relevant to 
        the model at hand, a reference to the relevant theoretical literature 
        is made.

        NOT YET IMPLEMENTED FOR THIS CLASS

        Parameters
        ----------
        verbose : boolean
            Specifies different levels of verbosity of feedback. When false, it 
            only reports whether the instance's type fails to satisfy a 
            particular condition. When true, it reports all results, i.e. the 
            factor values for all conditions.

        Returns
        -------
        None
        '''
        raise NotImplementedError()
        
class IRAPerfForesightConsumerType(HARKobject):
    '''
    Not the final product: short term consumer class for solving a lifecycle
    consumer model with an illiquid IRA and a liquid account, with no
    borrowing, and expiration of the penalty after a given period.
    '''
    def __init__(self,IncomeProfile,DiscFac,CRRA,Rsave,Rira,PenIRA,MaxIRA,
                 T_cycle,T_ira,InitialProblem):
        
        self.IncomeProfile      = IncomeProfile
        self.DiscFac            = DiscFac
        self.CRRA               = CRRA
        self.Rsave              = Rsave
        self.Rira               = Rira
        self.PenIRA             = PenIRA
        self.MaxIRA             = MaxIRA
        self.T_cycle            = T_cycle
        self.T_ira              = T_ira
        self.InitialProblem     = InitialProblem
    
    def solve(self):
        
        # Initialize the solution
        PenT = 0.0 if self.T_ira <= self.T_cycle else 0.1
        
        solution = [ConsIRAPFterminal(self.CRRA,PenT)]
        
        # Add rest of solutions
        for i in reversed(range(self.T_cycle-1)):
            if i >= self.T_ira - 1:
                solution.append(ConsIRAPFnoPen(self.IncomeProfile[i+1],
                                               self.DiscFac,self.CRRA,
                                               self.Rsave,self.Rira,
                                   self.MaxIRA,solution[self.T_cycle - i - 2]))
            
            elif ((i < self.T_ira - 1 and i > 0) or 
                  (i == 0 and not self.InitialProblem)):
                solution.append(ConsIRAPFpen(self.IncomeProfile[i+1],
                                             self.DiscFac,self.CRRA,
                                             self.Rsave,self.Rira,
                                             self.PenIRA,self.MaxIRA,
                                             solution[self.T_cycle - i - 2]))
            
            else:
                solution.append(ConsIRAPFinitial(self.IncomeProfile[i+1],
                                                 self.DiscFac,self.CRRA,
                                                 self.Rsave,self.Rira,
                                               solution[self.T_cycle - i - 2]))

        solution.reverse()        
        
        self.solution = solution
    
    def simulate(self,w0):
        
        self.w0 = w0
        
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
        
    def graphSim(self,saveFig=0,savePath='',graphLab =''):
        
        # create lifecycle arrays
        keys = ['aFunc','bFunc','cFunc','dFunc']
        
        a, b, c, d = [np.concatenate([k[ki] for k in self.simulation],axis=0) 
                                      for ki in keys]
        
        y = self.IncomeProfile
        y[0] = self.w0
        
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
        plt.xticks(tvar[1:])
        starter = np.round(np.min([c[1:],y[1:]]) -  (np.max([c[1:],y[1:]]) 
                           - np.min([c[1:],y[1:]]))/6, decimals = 2)
        stopper = np.round(np.max([c[1:],y[1:]]) -  (np.max([c[1:],y[1:]]) 
                           - np.min([c[1:],y[1:]]))/6, decimals = 2)
        plt.yticks(np.linspace(starter,stopper,6))
        plt.axvline(x=self.T_ira-1,color = 'C3')
        if saveFig:
            plt.savefig(savePath + '/IRAPFcons_' + graphLab + '.png')
        plt.show()
        
###############################################################################

def main():
    
    w0 = 0.25
    T = 8
    T_ira = 6
    y = np.array(T*[1])
    b = 1
    g = 2
    ra = 1
    r = 1.1
    dMax = .5
    t = .2
    
    
    IRAPF = IRAPerfForesightConsumerType(y,b,g,ra,r,t,dMax,T,T_ira,1)
    IRAPF.solve()
    IRAPF.simulate(w0)
    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P')
    
    y[4] = .75
    
    IRAPF = IRAPerfForesightConsumerType(y,b,g,ra,r,t,dMax,T,T_ira,1)
    IRAPF.solve()
    IRAPF.simulate(w0)
    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P5')
    
    y[4] = 1
    y[3] = .75
    
    IRAPF = IRAPerfForesightConsumerType(y,b,g,ra,r,t,dMax,T,T_ira,1)
    IRAPF.solve()
    IRAPF.simulate(w0)
    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P4')
    
    y[4] = 1
    y[3] = 1
    y[2] = .75
    
    IRAPF = IRAPerfForesightConsumerType(y,b,g,ra,r,t,dMax,T,T_ira,1)
    IRAPF.solve()
    IRAPF.simulate(w0)
    IRAPF.graphSim(saveFig=1,savePath='IRA_Results',graphLab='8P3')    
        
if __name__ == '__main__':
    main()