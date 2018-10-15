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
from scipy.optimize import basinhopping
from time import clock,time
from joblib import Parallel, delayed
import dill as pickle
import multiprocessing

import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

from core import AgentType, NullFunc, HARKobject
from interpolation import CubicInterp, LowerEnvelope, LinearInterp,\
                           BilinearInterp, ConstantFunction
from ConsIndShockModel import ConsumerSolution, ConsIndShockSolver, \
                                constructAssetsGrid
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
    
    def __init__(self, cFunc=None, dFunc=None, vFunc=None,
                       vPfunc=None, vPPfunc=None, mNrmMin=None, mNrmMin0=None,
                       nNrmMin=None, hNrm=None, MPCmin=None, MPCmax=None):
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
        vPPfunc : function
            The beginning-of-period marginal marginal value function, with 
            respect to m, for this period, defined over liquiud market 
            resources and illiquid account balance: vPP = vPPfunc(m,n)
        mNrmMin : function
            The minimum allowable liquid market resources for this period; 
            the consumption function (etc) are undefined for m < mNrmMin(n).
        mNrmMin0 : float
            The minimum allowable liquid market resources for this period,
            conditional on having zero illiquid assets
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
        if vPPfunc is None:
            vPPfunc = NullFunc()
        self.cFunc        = cFunc
        self.dFunc        = dFunc
        self.vFunc        = vFunc
        self.vPfunc       = vPfunc
        self.vPPfunc      = vPPfunc
        self.mNrmMin      = mNrmMin
        self.mNrmMin0     = mNrmMin0
        self.nNrmMin      = nNrmMin
        self.hNrm         = hNrm
        self.MPCmin       = MPCmin
        self.MPCmax       = MPCmax
        
class PureConsumptionFunc(HARKobject):
    '''
    A class for representing a pure consumption function.  The underlying 
    interpolation is in the space of (l,b). If b is degenerate, uses
    LinearInterp. If b is not degenerate, uses interp2d. When l <
    l_min(b), returns c = 0.
    '''
    distance_criteria = ['l_list','b_list','c_list']

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
        self.bZero = np.array(b_list == 0).all()
        self.lMin  = deepcopy(lMin)
        
        if self.bZero: # b grid is degenerate
            self.interpolator = LinearInterp(l_list,c_list,intercept_limit,
                                             slope_limit)
        else: # b grid is not degenerate
            self.interpolator = BilinearInterp(c_list,l_list,b_list)

    def __call__(self,l,b):
        '''
        Evaluate the pure consumption function at given levels of liquid 
        market resources l and illiquid assets b.

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
        assert np.array(b >= 0).all(), 'b should be non-negative'
        
        if self.bZero:
            c = self.interpolator(l)
        else:
            c = self.interpolator(l,b)
        
        # Set consumpgion to zero if l is below asset minimum
        c[l <= self.lMin(np.asarray(b))] = 0
        
        return c
    
class EndOfPeriodValueFunc(HARKobject):
    '''
    A class for representing the end-of-period value function, given end of 
    period assets a and b.  The underlying interpolation is in the space of 
    (a,b). If b is degenerate, uses LinearInterp. If b is not degenerate, uses 
    BilinearInterp.
    '''
    distance_criteria = ['a_list','b_list','w_list']

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
        self.bZero = np.array(b_list == 0).all()
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
        
        # Set w to u(0.0001) if m is below asset minimum
        w[a <= self.aMin(np.asarray(b))] = self.u(0.0001)
            
        return w
            
class ConsIRAPolicyFunc(HARKobject):
    '''
    A class for representing the optimal consumtion and deposit/withdrawal 
    functions.  The underlying interpolation is in the space of (m,n). If n is 
    degenerate, uses LinearInterp for consumption. If n is not degenerate, uses 
    interp2d for consumption and deposit/withdrawal. Always obeys:
        
        l = m - (1-t(d))*d
        b = n + d
        c = c(l,b)
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
        cFucnPure : float
            (Nomralized) consumption as a function of illiquid assets, l, and
            end-of-period illiquid assets, b.
            
        Returns
        -------
        None
        '''
        assert np.array(n_list >= 0).all(), 'n should be non-negative'
        self.nZero = np.array(n_list == 0).all()
        self.MaxIRA = MaxIRA
        self.PenIRA = PenIRA
        self.cFuncPure = deepcopy(cFuncPure)
        self.output = output
        
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
            d = 0
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
        
class ConsIRASolver(ConsIndShockSolver):
    '''
    A class for solving a single period of a consumption-savigs problem with
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
                     aXtraGrid,bXtraGrid,lXtraGrid,vFuncBool,CubicBool):
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
        DistIRA: float
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
                         'n': 'illiduid market resources at decisiont time',
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
        # Calculate PDV factor for illiquid assets next period when
        # a. account is liquidated next period
        if self.DistIRA > 1: # There is a penalty tomorrow
            bPDVFactorWithdrawNext = (1 - self.PenIRA)*(self.Rira/self.Rboro)
        else: # No penalty tomorrow
            bPDVFactorWithdrawNext = (self.Rira/self.Rboro)
        
        # b. account isn't liquidated until T_ira
        if self.DistIRA > 0:
            bPDVFactorWithdrawT_ira = (self.Rira/self.Rboro)**self.DistIRA
        else:
            bPDVFactorWithdrawT_ira = (self.Rira/self.Rboro)
        
        bPDVFactorWithdrawNow = (1 - self.PenIRA)
        
        # Take maximum PDV factor
        bPDVFactor = max(bPDVFactorWithdrawNext,bPDVFactorWithdrawT_ira)
        bPDVFactor_n = max(bPDVFactorWithdrawNow,bPDVFactorWithdrawT_ira)
        
        # Calculate the minimum allowable value of money resources in this 
        # period, when b = 0
        BoroCnstNat0 = ((self.solution_next.mNrmMin0 - self.TranShkMinNext)*
                           (self.PermGroFac*self.PermShkMinNext)/self.Rboro)
                           
        # Create natural borrowing constraint for different values of b
        self.BoroCnstNata = BoroCnstNat0 - np.append([0],bPDVFactor*np.asarray(
                        self.bXtraGrid))
        
        self.BoroCnstNatn = BoroCnstNat0 - np.append([0],bPDVFactor_n*
                                                     np.asarray(self.bXtraGrid)
                                                     )
                           
        # Note: need to be sure to handle BoroCnstArt==None appropriately. 
        # In Py2, this would evaluate to 5.0:  np.max([None, 5.0]).
        # However in Py3, this raises a TypeError. Thus here we need to 
        # directly address the situation in which BoroCnstArt == None:
        if BoroCnstArt is None:
            self.mNrmMin0 = BoroCnstNat0
            self.aNrmMinb = self.BoroCnstNata
            self.mNrmMinn = self.BoroCnstNatn
        else:
            self.mNrmMin0 = np.max([BoroCnstNat0,BoroCnstArt])
            self.aNrmMinb = np.maximum(BoroCnstArt,self.BoroCnstNata)
            self.mNrmMinn = np.maximum(BoroCnstArt,self.BoroCnstNatn)
            
        if BoroCnstNat0 < self.mNrmMin0: 
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

    def getPointsForPureConsumptionInterpolation(self,EndOfPrdv,
                                                 EndOfPrdvP,aNrmNow,bNrmNow):
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
        bNrmNow : np.array
            Array of end-of-period illiquid asset values that yield the
            marginal values in EndOfPrdvP.
        lXtraGrid : np.array
            Array of "extra" liquid assets just before the consumption decision
            -- assets above the abolute minimum acceptable level. 

        Returns
        -------
        c_for_interpolation : np.array
            Consumption points for interpolation. A flattened array of size 
            (lXtraGrid + 1) * bNrmNow.size.
        l_for_interpolation : np.array
            Corresponding liquid market resource points for interpolation of
            size (lXtraGrid + 1) * bNrmNow.size.
        b_for_interpolation : np.array
            Corresponding illiquid market resource points for interpolation of
            size (lXtraGrid + 1) * bNrmNow.size.
        '''
        cNrm_ik = self.uPinv(EndOfPrdvP)
        lNrm_ik = cNrm_ik + aNrmNow
        
        # Construct b-specific grids for l, including borrowing constraint
        # Then construct one grid for l, using non-overlapping segments of 
        # b-specific grids
        lNrm_jk = np.tile(np.insert(np.asarray(self.lXtraGrid),0,0.0),
                          (self.bNrmCount,1)) + np.transpose([self.aNrmMinb])
        lNrm_jk_Xtra = [lNrm_jk[i][lNrm_jk[i] < np.min(lNrm_jk[i-1])] for i in
                                range(1,len(lNrm_jk))]
        lNrm_j = np.sort(np.append(lNrm_jk[0],np.hstack(lNrm_jk_Xtra)))
        
        lNrmCount = lNrm_j.size
        
        # Construct b_k x l_j specific grids for l,c,a, and w
        lNrm_ik_temp,cNrm_ik_temp,aNrm_ik_temp,w_ik_temp = \
            [np.transpose(np.tile(x,(lNrmCount,1,1)),(1,0,2)) for x in 
             [lNrm_ik,cNrm_ik,aNrmNow,EndOfPrdv]]
        
        # Find where l_j in [l_ik , l_i+1k]
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
        
        cNrm_j_k = [[c[np.argmax(v)] if c.size > 0 else 0 for c,v in 
                    zip(ci,vi)] for ci,vi in zip(cNrm_j_ik,v_j_ik)]
            
        c_for_interpolation = np.transpose(np.array(cNrm_j_k))
        l_for_interpolation = lNrm_j
        b_for_interpolation = bNrmNow
        
        return c_for_interpolation, l_for_interpolation, b_for_interpolation
    
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
        
        self.EndOfPrdvFunc = EndOfPeriodValueFunc(aNrmNowUniform,self.bNrmNow,
                                                  EndOfPrdv_trans,
                                                  self.BoroCnstFunc,self.u)
        self.aNrmNowUniform = aNrmNowUniform
        
    def makevOfdFunc(self,dNrm,mNrm,nNrm):
        '''
        Constructs a beginning-period value function, given the IRA deposit (d)
        , beginning-of-period liquid resources and beginning-of-period illiquid
        assets. Since a minimizer is used, returns negative of the value
        function.
        
        Parameters
        ----------
        dNrm : float
            (Normalized) IRA deposit/withdrawal this period.
        mNrm : float
            (Normalized) liquid assets at the beginning of this period.
        nNrm : float
            (Normalized) illiquid assets at the beginning of this period.
        
        Returns
        -------
        v : float
            Negative 1 times the value function given d, m, and n.
        '''
        bNrm = nNrm + dNrm
        assert np.array(bNrm >= 0).all(), 'b should be non-negative, values' + str(dNrm) + ' ' + str(mNrm) + ' ' + str(nNrm) + ' .'
        
        lNrm = mNrm - (1 - self.PenIRA*(dNrm < 0))*dNrm
        cNrm = self.cFuncNowPure(lNrm,bNrm)
        aNrm = lNrm - cNrm
        
        # can't actually evaluate cNrm == 0
        if not cNrm > 0.0:
            cNrm = 0.0001
        
        v = self.u(cNrm) + self.EndOfPrdvFunc(aNrm,bNrm)
        
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
        d = basinhopping(self.makevOfdFunc,
                         0,minimizer_kwargs={"bounds":((-nNrm + 1e-10,
                                                        self.MaxIRA),),
                                             "args":(mNrm,nNrm)}).x
        
        return d
        
    def makecAnddFunc(self):
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
            self.dFuncNow = ConstantFunction(0)
            self.cFuncNow = self.cFuncNowPure
        else:
            mNrm = self.aNrmNowUniform
            nNrm = self.bNrmNow
            
            n_cpus = multiprocessing.cpu_count()
           
            dNrm_list = Parallel(n_jobs=n_cpus)(delayed(self.findArgMaxv)(m,n) 
                                                for n in nNrm for m in mNrm)
            
            dNrm = np.asarray(dNrm_list).reshape(len(nNrm),len(mNrm))
            dNrm_trans = np.transpose(dNrm)
            
            self.cFuncNow = ConsIRAPolicyFunc(mNrm,nNrm,dNrm_trans,self.MaxIRA,
                                              self.PenIRA,self.cFuncNowPure,
                                              output='cFunc')
            self.dFuncNow = ConsIRAPolicyFunc(mNrm,nNrm,dNrm_trans,self.MaxIRA,
                                              self.PenIRA,self.cFuncNowPure,
                                              output='dFunc')
    

        
        
            
        