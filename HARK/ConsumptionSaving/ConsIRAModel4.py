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

def unwrap_self(arg, **kwarg):
    '''
    Auxiliary function needed in order to run the multiprocessing command Pool
    within a method of a class below. This gets around Pool having to call a
    method, i.e. self.findArgMaxv. Multiprocessing needs functions that can be 
    called in a global context, in order to "pickle."
    '''
    return ConsIRASolver.findArgMaxv(*arg, **kwarg)

aNrmNowUniform = np.array([.1,.5,1,2,3,4])
bNrmNow = np.array([.5,1,2,3])

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
    
    def makeAB(self):
        self.aNrmNowUniform = np.asarray(self.aXtraGrid)
        self.bNrmNow = np.asarray(self.bXtraGrid)
    
    def u(self,c):
        return utility(c,gam=self.CRRA)
    
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
        
        v = self.u(cNrm) + self.makeEndOfPrdvFunc(aNrm,bNrm)
        
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
        mNrm = self.aNrmNowUniform
        nNrm = self.bNrmNow
            
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

# =============================================================================
# ================ Other useful functions =====================================
# =============================================================================


###############################################################################

def main():
    import ConsIRAParameters as Params
    mystr = lambda number : "{:.4f}".format(number)

    # Make and solve an example IRA consumer
    IRASolverExample = ConsIRASolver(**Params.init_IRA_Solver)
    IRASolverExample.makeAB()
    
    start_time = clock()
    start_time2 = time()
    IRASolverExample.makePolicyFunc()
    end_time = clock()
    end_time2 = time()
    print('Solving an IRA consumer took ' + mystr(end_time-start_time/3600) +\
          ' processor hours.')
    print('Solving an IRA consumer took ' + mystr(end_time2-start_time2/3600)+\
          ' real hours.')
    
    

    pickle.dump_session('IRAex.pkl')
        
if __name__ == '__main__':
    main()