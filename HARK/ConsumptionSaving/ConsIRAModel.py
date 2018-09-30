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
from scipy.optimize import minimize_scalar

import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

from core import AgentType, NullFunc, HARKobject
from interpolation import CubicInterp, LowerEnvelope, LinearInterp,\
                           BilinearInterp
from ConsIndShockModel import ConsumerSolution, ConsIndShockSolver
from simulation import drawDiscrete, drawBernoulli, drawLognormal, drawUniform
from utilities import approxMeanOneLognormal, addDiscreteOutcomeConstantMean,\
                           combineIndepDstns, makeGridExpMult, CRRAutility, \
                           CRRAutilityP, CRRAutilityPP, CRRAutilityP_inv, \
                           CRRAutility_invP, CRRAutility_inv, CRRAutilityP_invP 

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
        mNrmMin : float
            The minimum allowable liquid market resources for this period; 
            the consumption function (etc) are undefined for m < mNrmMin.
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
    LinearInterp. If b is not degenerate, uses BilinearInterp.
    '''
    distance_criteria = ['c_array','l_list','b_list']

    def __init__(self,c_array,l_list,b_list,intercept_limit=None,
                 slope_limit=None):
        '''
        Constructor for a pure consumption function, c(l,b).

        Parameters
        ----------
        cNrm : np.array
            (Normalized) consumption points for interpolation. If b is
            degenerate, this is a 1D array, otherwise it's a 2D array.
        lNrm : np.array
            (Normalized) grid of liquid market resource points for 
            interpolation.
        bNrm : np.array
            (Normalized) grid of illiquid market resource points for 
            interpolation.
        interpolator : function
            A function that constructs and returns a consumption function,
            either a 2D or 1D interpolator.
            
        Returns
        -------
        None
        '''
        self.cNrm                = cNrm
        self.lNrm                = lNrm
        self.bNrm                = bNrm
        self.intercept_limit     = intercept_limit
        self.slope_limit         = slope_limit
        self.bZero               = bNrm.size == 1
        
        if self.bZero: # b grid is degenerate
            self.interpolator = LinearInterp(cNrm,lNrm,intercept_limit,
                                             slope_limit)
        else: # b grid is not degenerate
            self.interpolator = BilinearInterp(cNrm,lNrm,bNrm)

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
        if self.bZero:
            assert np.sum(b) == 0, 'Illiquid assets shoudl be zero!'
            c = self.interpolator(l)
        else:
            c = self.interpolator(l,b)
            
        return c
        
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
        assert Rboro>=Rira>=Rsave, 'Interest factors must satisfy \
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
        # kinked-R basic case, so start with that.
        ConsIndShockSolver.__init__(self,solution_next,IncomeDstn,LivPrb,
                                   DiscFac,CRRA,Rsave,PermGroFac,BoroCnstArt,
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
        
        # Take maximum PDV factor
        bPDVFactor = max(bPDVFactorWithdrawNext,bPDVFactorWithdrawT_ira)
        
        # Calculate the minimum allowable value of money resources in this 
        # period, when b = 0
        BoroCnstNat0 = ((self.solution_next.mNrmMin0 - self.TranShkMinNext)*
                           (self.PermGroFac*self.PermShkMinNext)/self.Rfree)
                           
        # Create natural borrowing constraint for different values of b
        self.BoroCnstNat = BoroCnstNat0 - np.append([0],bPDVFactor*np.asarray(
                        self.bXtraGrid))
                           
        # Note: need to be sure to handle BoroCnstArt==None appropriately. 
        # In Py2, this would evaluate to 5.0:  np.max([None, 5.0]).
        # However in Py3, this raises a TypeError. Thus here we need to 
        # directly address the situation in which BoroCnstArt == None:
        if BoroCnstArt is None:
            self.mNrmMin0 = BoroCnstNat0
            self.aNrmMinb = self.BoroCnstNat
        else:
            self.mNrmMin0 = np.max([BoroCnstNat0,BoroCnstArt])
            self.aNrmMinb = np.maximum(BoroCnstArt,self.BoroCnstNat)
            
        if BoroCnstNat0 < self.mNrmMin0: 
            self.MPCmaxEff = 1.0 # If actually constrained, MPC near limit is 1
        else:
            self.MPCmaxEff = self.MPCmaxNow
    
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
        if self.bXtragrid.size == 0:
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
        return aNrmNow, bNrmNow

    def calcEndOfPrdvAndvP(self):
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
        sum_axis = self.mNrmNext.ndim - 2
        
        EndOfPrdv   = self.DiscFacEff*\
                            np.sum(self.PermShkVals_temp**
                                   (1.0-self.CRRA)*self.PermGroFac**
                                   (1.0-self.CRRA)*
                                   self.vFuncNext(self.mNrmNext,self.nNrmNext)*
                                   self.ShkPrbs_temp,axis=sum_axis)
        
        EndOfPrdvP  = self.DiscFacEff*\
                            self.Rfree_Mat*\
                            self.PermGroFac**(-self.CRRA)*\
                            np.sum(self.PermShkVals_temp**(-self.CRRA)*\
                                   self.vPfuncNext(self.mNrmNext,self.nNrmNext)
                                   *self.ShkPrbs_temp,axis=sum_axis)
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
            Consumption points for interpolation, of shape (ln,bn).
        l_for_interpolation : np.array
            Corresponding liquid market resource points for interpolation of
            size ln.
        b_for_interpolation : np.array
            Corresponding illiquid market resource points for interpolation of
            size bn.
        '''
        cNrm_ik = self.uPinv(EndOfPrdvP)
        lNrm_ik = cNrm_ik + aNrmNow
        
        # Construct b-specific grids for l, including borrowing constraint
        # Then construct one grid for l, using non-overlapping segments of 
        # b-specific grids
        if self.bNrmCount > 1:
            lNrm_jk = np.tile(np.insert(np.asarray(self.lXtraGrid),0,0.0),
                          (self.bNrmCount,1)) + np.transpose([self.aNrmMinb])
            lNrm_jk_Xtra = [lNrm_jk[i][lNrm_jk[i] < np.min(lNrm_jk[i-1])] for 
                                    i in [1,len(lNrm_jk)-1]]
            lNrm_j = np.sort(np.append([lNrm_jk[0],
                                    np.hstack(lNrm_jk_Xtra)]))
        else:
            lNrm_j = np.insert(np.asarray(self.lXtraGrid),0,0.0) \
                        + self.aNrmMinb
        
        lNrmCount = lNrm_j.size
        
        # Find where l_j in [l_ik , l_i+1k]
        lNrm_j_temp = np.tile(lNrm_j[:,None],(1,self.bNrmCount))[:,:,None]
        lNrm_ik_temp = np.tile(lNrm_ik,(lNrmCount,1,1))
        
        lNrm_j_mask = (lNrm_j_temp > lNrm_ik_temp[:,:,:-1]) \
                        & ~(lNrm_j_temp > lNrm_ik_temp[:,:,1:])
        
        
        i = [[np.flatnonzero(row) for row in mat] for mat in lNrm_j_mask]
        
        # Calculate candidate optimal consumption, c_j_ik
        # Calculate associated assets, a_j_ik, and next period value, w_j_ik
        # Find consumption that maximizes utility
        cNrm_ik_temp = np.tile(cNrm_ik,(lNrmCount,1,1))
        aNrm_ik_temp = np.tile(aNrmNow,(lNrmCount,1,1))
        w_ik_temp = np.tile(EndOfPrdv,(lNrmCount,1,1))
        
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
        
        if self.bNrmCount > 1:
            c_for_interpolation = np.array(cNrm_j_k)
        else:
            c_for_interpolation = np.array(cNrm_j_k).flatten()
            
        l_for_interpolation = lNrm_j
        b_for_interpolation = bNrmNow
        
        return c_for_interpolation, l_for_interpolation, b_for_interpolation
    
    def usePointsForPureConsumptionInterpolation(self,cNrm,lNrm,bNrm):
        '''
        Constructs a pure consumption function c(l,b), i.e. one consumption
        given l, holding b fixed this period (no deposits or withdrawals).

        Parameters
        ----------
        cNrm : np.array
            (Normalized) consumption points for interpolation. If b is
            degenerate, this is a 1D array, otherwise it's a 2D array.
        lNrm : np.array
            (Normalized) grid of liquid market resource points for 
            interpolation.
        bNrm : np.array
            (Normalized) grid of illiquid market resource points for 
            interpolation.
        interpolator : function
            A function that constructs and returns a consumption function,
            either a 2D or 1D interpolator.

        Returns
        -------
        purecFuncNow : LinearInterp or BilinearInterp
            The pure consumption function for this period.
        '''
        if self.bNrmCount == 1:
            purecFuncNow = PureConsumptionFunc(cNrm,lNrm,cNrm,
                                               self.MPCminNow*self.hNrmNow,
                                               self.MPCminNow)
        else:
            purecFuncNow = PureConsumptionFunc(cNrm,lNrm,bNrm)
        
        return purecFuncNow