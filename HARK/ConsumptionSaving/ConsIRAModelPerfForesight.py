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
from time import clock, time
import multiprocessing as mp
from pathos.multiprocessing import ProcessPool

import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

import matplotlib.pyplot as plt

from core import NullFunc, HARKobject
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
    
class TerminalMargValueFunc2D(HARKobject):
    '''
    A class for representing a the marginal terminal value with respect to
    consumption, in a model with liquid assets and illiquid assets.
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
        return utilityP(self.cFunc(m,n),gam=self.CRRA)
    
class ConsIRAPFSolution(HARKobject):
    '''
    A class representing the perfect foresight solution of a single period of a 
    consumption-saving model with a liquid account and an IRA-like illiquid
    savings account. The solution must include a consumption function, an
    optimal illiquid deposit function, value function, marginal value 
    function with respect to the illiquid asset, and marginal value function
    with respect to the liquid asset. One additional function, regimeFunc, keeps
    track of whether we are at a corner, kink, or interior solution.

    In the perfect foresight model, 
    '''
    distance_criteria = ['cFunc','dFunc']
    
    def __init__(self, cFunc=None, dFunc=None, policyFunc=None, vFunc=None, 
                 vPmFunc=None, vPnFunc=None, regimeFunc=None):
        '''
        The constructor for a new ConsumerIRAPFSolution object.

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
        vPmFunc : function
            The beginning-of-period marginal value function, with respect to
            m, for this period, defined over liquiud market resources and 
            illiquid account balance: vP = dvfunc(m,n)/dm
        vPnFunc : function
            The beginning-of-period marginal value function, with respect to
            n, for this period, defined over liquiud market resources and 
            illiquid account balance: vP = dvfunc(m,n)/dn
        regimeFunc : function
            Keeps track of whether the consumer is a corner, kink, or intertior
            solution, as a function of m and n

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
        if vPmFunc is None:
            vPmFunc = NullFunc()
        if vPnFunc is None:
            vPnFunc = NullFunc()
        if regimeFunc is None:
            regimeFunc = NullFunc()
        self.cFunc        = cFunc
        self.dFunc        = dFunc
        self.policyFunc   = policyFunc
        self.vFunc        = vFunc
        self.vPmFunc      = vPmFunc
        self.vPnFunc      = vPnFunc
        
# ====================================
# === Perfect foresight IRA model ===
# ====================================
        
class ConsIRAPFSolver(HARKobject):
    '''
    A class for solving a period of a perfect foresight consumption-saving 
    problem, with an illiquid and liquid account. 
    '''
        
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
            bgrid = HARKobject
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
        
###############################################################################

def main():
    import ConsIRAParameters as Params
    #from ConsIndShockModel import KinkedRconsumerType
    
    mystr = lambda number : "{:.4f}".format(number)

    do_simulation = True
    
    # Make and solve an example IRA consumer
    IRAexample = IRAConsumerType(**Params.init_IRA_30_simp)
    IRAexample.cycles = 1 # Make this consumer live a sequence of periods
                          # exactly once
                          
    # Extend the recursion depth limit
    recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)
    
    start_time = clock()
    start_time2 = time()
    IRAexample.solve()
    end_time = clock()
    end_time2 = time()
    print('Solving an IRA consumer took ' + mystr((end_time-start_time)/3600)+\
          ' processor hours.')
    print('Solving an IRA consumer took ' +\
          mystr((end_time2-start_time2)/3600)+ ' real hours.')
    
    # Return to previous recursion depth limit
    sys.setrecursionlimit(recursion_limit)
    
    IRAexample.timeFwd()
    
    # Make and solve a 30 period kinked consumer
    KinkedExample = KinkedRconsumerType(**Params.init_IRA_30)
    KinkedExample.cycles = 1 # Make this consumer live a sequence of periods
                             # exactly once
                             
    start_time = clock()
    start_time2 = time()
    KinkedExample.solve()
    end_time = clock()
    end_time2 = time()
    print('Solving a Kinked consumer took ' + mystr((end_time-start_time)/3600)+\
          ' processor hours.')
    print('Solving a Kinked consumer took ' +\
          mystr((end_time2-start_time2)/3600)+ ' real hours.')

    KinkedExample.timeFwd()
    
    # Compare Consumption Functions
    
    # Get consumption function in periods 15, 20, 25
    mRange = {}
    cKinked = {}
    mRange['15'] = np.arange(KinkedExample.solution[15].mNrmMin,KinkedExample.solution[15].mNrmMin+3,.01)
    cKinked['15'] = KinkedExample.solution[15].cFunc(mRange['15'])
    mRange['20'] = np.arange(KinkedExample.solution[20].mNrmMin,KinkedExample.solution[20].mNrmMin+3,.01)
    cKinked['20'] = KinkedExample.solution[20].cFunc(mRange['20'])
    mRange['25'] = np.arange(KinkedExample.solution[25].mNrmMin,KinkedExample.solution[25].mNrmMin+3,.01)
    cKinked['25'] = KinkedExample.solution[25].cFunc(mRange['25'])
    
    # Get consumption function in period 15, 20, 25
    cIRA = {}
    cIRA['15'] = IRAexample.solution[15].cFunc(mRange['15'],np.zeros(mRange['15'].size))
    cIRA['20'] = IRAexample.solution[20].cFunc(mRange['20'],np.zeros(mRange['20'].size))
    cIRA['25'] = IRAexample.solution[25].cFunc(mRange['25'],np.zeros(mRange['25'].size))
    
    # Export consumption functions for Kinked and IRA consumers
    data15 = np.array([mRange['15'].T,cKinked['15'].T,cIRA['15'].T,15*np.ones(mRange['15'].size).T])
    data20 = np.array([mRange['20'].T,cKinked['20'].T,cIRA['20'].T,20*np.ones(mRange['20'].size).T])
    data25 = np.array([mRange['25'].T,cKinked['25'].T,cIRA['25'].T,25*np.ones(mRange['25'].size).T])
    
    data = np.concatenate((data15.T,data20.T,data25.T))
    
    np.savetxt('IRA_Results/IRA_Kinked_data.csv',data,delimiter=',',header='mRange,cKinked,cIRA,period')
    
    # Plot the consumption functions beside each other
    
    def comparePlots(period):
        x = mRange[str(period)]
        y1 = cKinked[str(period)]
        y2 = cIRA[str(period)]
        plt.plot(x,y1,'C1',label='Kinked Consumer')
        plt.plot(x,y2,'C0--',label='IRA Consumer')
        plt.xlabel('liquid assets')
        plt.ylabel('consumption')
        plt.title('Consumption Functions: Period ' + str(period))
        plt.legend()
        plt.grid()
        plt.savefig('IRA_Results/IRA_Kinked_' + str(period) + '.png')
        plt.show()
        
    comparePlots(15)
    comparePlots(20)
    comparePlots(25)
    
###############################################################################

# Run Simulations & Plot Time Series
    

    if do_simulation:
        IRAexample.T_sim = 120
        IRAexample.track_vars = ['aNrmNow','bNrmNow','mNrmNow','nNrmNow',
                                 'cNrmNow','dNrmNow','pLvlNow','t_age']
        IRAexample.initializeSim()
        IRAexample.simulate()
    
    np.savetxt('IRA_Results/a_30_comp.csv',IRAexample.aNrmNow_hist.T,delimiter=',')
    np.savetxt('IRA_Results/b_30_comp.csv',IRAexample.bNrmNow_hist.T,delimiter=',')
    np.savetxt('IRA_Results/m_30_comp.csv',IRAexample.mNrmNow_hist.T,delimiter=',')
    np.savetxt('IRA_Results/n_30_comp.csv',IRAexample.nNrmNow_hist.T,delimiter=',')
    np.savetxt('IRA_Results/c_30_comp.csv',IRAexample.cNrmNow_hist.T,delimiter=',')
    np.savetxt('IRA_Results/d_30_comp.csv',IRAexample.dNrmNow_hist.T,delimiter=',')
    np.savetxt('IRA_Results/p_30_comp.csv',IRAexample.pLvlNow_hist.T,delimiter=',')
    np.savetxt('IRA_Results/t_30_comp.csv',IRAexample.t_age_hist.T,delimiter=',')
    print('Data From Simulations Exported')
    
    if do_simulation:
        KinkedExample.T_sim = 120
        KinkedExample.track_vars = ['aNrmNow','mNrmNow','cNrmNow',
                                    'pLvlNow','t_age']
        KinkedExample.initializeSim()
        KinkedExample.simulate()
        
    np.savetxt('IRA_Results/a_30_kink.csv',KinkedExample.aNrmNow_hist.T,delimiter=',')
    np.savetxt('IRA_Results/m_30_kink.csv',KinkedExample.mNrmNow_hist.T,delimiter=',')
    np.savetxt('IRA_Results/c_30_kink.csv',KinkedExample.cNrmNow_hist.T,delimiter=',')
    np.savetxt('IRA_Results/p_30_kink.csv',KinkedExample.pLvlNow_hist.T,delimiter=',')
    np.savetxt('IRA_Results/t_30_kink.csv',KinkedExample.t_age_hist.T,delimiter=',')
    print('Data From Simulations Exported')
    
    # Plot Time Series from Simulations
    def averageSimulations(plot_list,age_array,p_array,period_T):
        
        def collapse_var(x,age,p,t):
            inflated = x[age == t]*p[age == t]
            averaged = np.average(inflated)
            return averaged
        
        averagePlot = [[collapse_var(j,age_array,p_array,i) 
                        for i in range(1,period_T+1)] for j in plot_list]
        
        return averagePlot
    
    # Plot Kinked Time Series
    
    kinked_plot_list = [KinkedExample.mNrmNow_hist,
                        KinkedExample.cNrmNow_hist,
                        KinkedExample.aNrmNow_hist]
    
    period_T_kink = np.amax(KinkedExample.t_age_hist)
    
    m_kinked,c_kinked,a_kinked = averageSimulations(kinked_plot_list,
                                                    KinkedExample.t_age_hist,
                                                    KinkedExample.pLvlNow_hist,
                                                    period_T_kink)
    
    age_kink = np.arange(1,period_T_kink+1)
    
    # plot average assets and consumption
    plt.plot(age_kink,m_kinked, label = 'beginning of period assets')
    plt.plot(age_kink,a_kinked, label = 'end of period assets')
    plt.plot(age_kink,c_kinked, label = 'consumption')
    plt.xlabel('age')
    plt.ylabel('level')
    plt.title('Kinked Consumer: Times Series')
    plt.grid()
    plt.legend()
    plt.savefig('IRA_Results/KinkedTimeSeries_vs.png')
    plt.show()
    
    # Plot IRA Time Series
    
    ira_plot_list = [IRAexample.mNrmNow_hist,
                     IRAexample.nNrmNow_hist,
                     IRAexample.cNrmNow_hist,
                     IRAexample.dNrmNow_hist,
                     IRAexample.aNrmNow_hist,
                     IRAexample.bNrmNow_hist,]
    
    period_T_ira = np.amax(IRAexample.t_age_hist)
    
    m_ira,n_ira,c_ira,d_ira,a_ira,b_ira = averageSimulations(ira_plot_list,
                                                    IRAexample.t_age_hist,
                                                    IRAexample.pLvlNow_hist,
                                                    period_T_ira)
    
    age_ira = np.arange(1,period_T_ira+1)
    
    # plot average assets and consumption
    plt.plot(age_ira,m_ira, label = 'beginning of period assets')
    plt.plot(age_ira,a_ira, label = 'end of period assets')
    plt.plot(age_ira,c_ira, label = 'consumption')
    plt.xlabel('age')
    plt.ylabel('level')
    plt.title('IRA Consumer: Times Series')
    plt.grid()
    plt.legend()
    plt.savefig('IRA_Results/IRATimeSeries_vs.png')
    plt.show()
    
    # plot deposits/withdrawals and illiquid balance (should be zero)
    plt.plot(age_ira,n_ira, label = 'beginning of period assets')
    plt.plot(age_ira,b_ira, label = 'end of period assets')
    plt.plot(age_ira,d_ira, label = 'deposits/withdrawals')
    plt.xlabel('age')
    plt.ylabel('level')
    plt.title('IRA Consumer: Deposit Times Series')
    plt.grid()
    plt.legend()
    plt.savefig('IRA_Results/IRADepositTimeSeries_vs.png')
    plt.show()
        
if __name__ == '__main__':
    main()