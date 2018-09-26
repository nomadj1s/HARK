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
from interpolation import CubicInterp, LowerEnvelope, LinearInterp
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
                     Rira,PenIRA,DistIRA,PermGroFac,BoroCnstArt,aXtraGrid,
                     bXtraGrid,lXtraGrid,vFuncBool,CubicBool):
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
        
        # Assign factors, illiquid asset grid, and IRA penalty.
        self.Rboro        = Rboro
        self.Rsave        = Rsave
        self.Rira         = Rira
        self.PenIRA       = PenIRA
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
        bNrmCount   = np.asarray(self.bXtraGrid).size + 1
        aNrmCount   = np.asarray(self.aXtraGrid).size
        bNrmNow     = np.tile(np.insert(np.asarray(self.bXtraGrid),0,0.0)[:, 
                              np.newaxis],(1,aNrmCount))
        aNrmNow     = np.tile(np.asarray(self.aXtraGrid),(bNrmCount,1)) \
                        + np.transpose([self.aNrmMinb])
                 
        ShkCount    = self.TranShkValsNext.size
        aNrm_temp   = np.transpose(np.tile(aNrmNow,(ShkCount,1,1)),(1,0,2))
        bNrm_temp   = np.transpose(np.tile(bNrmNow,(ShkCount,1,1)),(1,0,2))

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
        Rfree_Mat[aNrmNow < 0] = self.Rboro
            
        # Get liquid assets next period
        mNrmNext   = Rfree_Mat[:, np.newaxis]/(self.PermGroFac*
                              PermShkVals_temp)*aNrm_temp + TranShkVals_temp
                            
        # Get illiquid assets nex period
        nNrmNext   = self.Rira/(self.PermGroFac*PermShkVals_temp)*bNrm_temp
        
        # If bXtragrid = [], remove extraneous dimension from arrays
        if self.bXtragrid.size == 0:
            for x in [aNrmNow,bNrmNow,nNrmNext,mNrmNext,PermShkVals_temp,
                      ShkPrbs_temp,TranShkVals_temp,Rfree_Mat]:
             x = x[0]

        # Store and report the results
        self.Rfree_Mat         = Rfree_Mat
        self.PermShkVals_temp  = PermShkVals_temp
        self.ShkPrbs_temp      = ShkPrbs_temp
        self.mNrmNext          = mNrmNext
        self.nNrmNext          = nNrmNext
        self.aNrmNow           = aNrmNow
        self.bNrmNow           = bNrmNow
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
                                                 EndOfPrdvP,aNrmNow,bNrmNow,
                                                 lXtraGrid):
        '''
        Finds interpolation points (c,l,b) for the pure consumption function.
        
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
        lNrmNow : np.array
            Array of 

        Returns
        -------
        c_for_interpolation : np.array
            Consumption points for interpolation.
        l_for_interpolation : np.array
            Corresponding liquid market resource points for interpolation.
        b_for_interpolation : np.array
            Corresponding illiquid market resource points for interpolation.
        '''
        cNrm_ik = self.uPinv(EndOfPrdvP)
        lNrm_ik = cNrm_ik + aNrmNow
        
        