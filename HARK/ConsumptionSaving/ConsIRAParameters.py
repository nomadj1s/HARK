'''
Specifies examples of the full set of parameters required to solve the 
ConsIRAModel.
'''
from __future__ import division, print_function
import numpy as np
from copy import copy

import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

from ConsIRAModel import ConsIRASolution, utility, utilityP

# -----------------------------------------------------------------------------
# --- Define all of the parameters for the consumption IRA model   ------------
# -----------------------------------------------------------------------------

CRRA = 2.0                          # Coefficient of relative risk aversion
DiscFac = 0.96                      # Intertemporal discount factor
                                    # Survival probability
LivPrb = [0.99,0.9,0.9,0.8,0.8,0.7,0.6,0.4,0.2,0.1]
# LivPrb = [liv for liv in LivPrb_10 for i in range(4)]

AgentCount = 10000                  # Number of agents of this type (only 
                                    # matters for simulation)
aNrmInitMean = 0.0                  # Mean of log initial assets (only matters 
                                    # for simulation)
aNrmInitStd  = 1.0                  # Standard deviation of log initial assets 
                                    # (only for simulation)
pLvlInitMean = 0.0                  # Mean of log initial permanent income 
                                    # (only matters for simulation)
pLvlInitStd  = 0.0                  # Standard deviation of log initial 
                                    # permanent income (only matters for 
                                    # simulation)
PermGroFacAgg = 1.0                 # Aggregate permanent income growth factor 
                                    # (only matters for simulation)
T_age = 11                          # Age after which simulated agents are 
                                    # automatically killed
T_cycle = 10                        # Number of periods in the cycle for this 
                                    # agent type

# Parameters for constructing the "assets above minimum" grid
aXtraMin = 0.001                    # Minimum end-of-period "assets above 
                                    # minimum" value
aXtraMax = 20                       # Maximum end-of-period "assets above 
                                    # minimum" value
aXtraExtra = None                   # Some other value of "assets above 
                                    # minimum" to add to the grid, not used
aXtraNestFac = 3                    # Exponential nesting factor when 
                                    # constructing "assets above minimum" grid
aXtraCount = 36                     # Number of points in the grid of "assets 
                                    # above minimum"

bXtraMin = 0.001                    # Minimum end-of-period "assets above 
                                    # minimum" value
bXtraMax = 20                       # Maximum end-of-period "assets above 
                                    # minimum" value
bXtraExtra = None                   # Some other value of "assets above 
                                    # minimum" to add to the grid, not used
bXtraNestFac = 3                    # Exponential nesting factor when 
                                    # constructing "assets above minimum" grid
bXtraCount = 36                     # Number of points in the grid of "assets 
                                    # above minimum"

# Parameters describing the income process
PermShkCount = 7                    # Number of points in discrete 
                                    # approximation to permanent income shocks
TranShkCount = 7                    # Number of points in discrete 
                                    # approximation to transitory income shocks
                                    # Permanent income growth factor
PermGroFac = [1.01,1.01,1.01,1.01,1.01,1.02,1.02,1.02,1.02,1.02]
                                    # Standard deviation of log permanent 
                                    # income shocks
PermShkStd = [0.1,0.2,0.1,0.2,0.1,0.2,0.1,0.2,0,0]                  
                                    # Standard deviation of log transitory 
                                    # income shocks
TranShkStd = [0.3,0.2,0.1,0.3,0.2,0.1,0.3,0.2,0,0]
UnempPrb = 0.05                     # Probability of unemployment while working
UnempPrbRet = 0.005                 # Probability of "unemployment" while 
                                    # retired
IncUnemp = 0.3                      # Unemployment benefits replacement rate
IncUnempRet = 0.0                   # "Unemployment" benefits when retired
tax_rate = 0.0                      # Flat income tax rate
T_retire = 9                        # Period of retirement (0 --> no 
                                    # retirement)

# A few other parameters
BoroCnstArt = None                  # Artificial borrowing constraint; imposed 
                                    # minimum level of end-of period assets
CubicBool = False                   # Use cubic spline interpolation when True,
                                    # linear interpolation when False
vFuncBool = True                    # Whether to calculate the value function 
                                    # during solution
ParallelBool = True                 # Whether to use multiprocessing or not

# Interest rate/factors

Rboro = 1.20                        # Interest factor on assets when borrowing,
                                    # a < 0
Rsave = 1.02                        # Interest factor on assets when saving, a 
                                    # > 0
Rira = 1.10                         # Interest factor on illiquid (IRA) assets

# Additional IRA parameters
T_ira = 7                           # Period of IRA penalty expiration
PenIRAFixed = 0.1                   # Penalty for early IRA withdrawals
MaxIRA = 0.2                        # Max deposit, normalized by permanent
                                    # income

# Make a dictionary to specify an IRA consumer type
init_IRA_10 = { 'CRRA': CRRA,
             'Rsave' : Rsave,
             'Rboro' : Rboro,
             'Rira' : Rira,
             'DiscFac': DiscFac,
             'LivPrb': LivPrb,
             'PermGroFac': PermGroFac,
             'AgentCount': AgentCount,
             'aXtraMin': aXtraMin,
             'aXtraMax': aXtraMax,
             'aXtraNestFac':aXtraNestFac,
             'aXtraCount': aXtraCount,
             'aXtraExtra': [aXtraExtra],
             'bXtraMin': bXtraMin,
             'bXtraMax': bXtraMax,
             'bXtraNestFac':bXtraNestFac,
             'bXtraCount': bXtraCount,
             'bXtraExtra': [bXtraExtra],
             'PermShkStd': PermShkStd,
             'PermShkCount': PermShkCount,
             'TranShkStd': TranShkStd,
             'TranShkCount': TranShkCount,
             'UnempPrb': UnempPrb,
             'UnempPrbRet': UnempPrbRet,
             'IncUnemp': IncUnemp,
             'IncUnempRet': IncUnempRet,
             'BoroCnstArt': BoroCnstArt,
             'tax_rate': tax_rate,
             'vFuncBool':vFuncBool,
             'CubicBool':CubicBool,
             'ParallelBool':ParallelBool,
             'T_retire':T_retire,
             'aNrmInitMean' : aNrmInitMean,
             'aNrmInitStd' : aNrmInitStd,
             'pLvlInitMean' : pLvlInitMean,
             'pLvlInitStd' : pLvlInitStd,
             'PermGroFacAgg' : PermGroFacAgg,
             'T_age' : T_age,
             'T_cycle' : T_cycle,
             'T_ira' : T_ira,
             'PenIRAFixed' : PenIRAFixed,
             'MaxIRA' : MaxIRA
            }

# -----------------------------------------------------------------------------
# --- Define all of the parameters for the 30 period consumption IRA model ----
# -----------------------------------------------------------------------------

init_IRA_30 = copy(init_IRA_10)

init_IRA_30['LivPrb'] = 30*[1]
init_IRA_30['T_age'] = 31
init_IRA_30['T_cycle'] = 30
init_IRA_30['PermGroFac'] = 30*[1.01]
init_IRA_30['PermShkStd'] = 26*[0.15] + 4*[0]
init_IRA_30['TranShkStd'] = 26*[0.2] + 4*[0]
init_IRA_30['T_retire'] = 26
init_IRA_30['T_ira'] = 22


# -----------------------------------------------------------------------------
# --- Define all of the parameters for the 40 period consumption IRA model ----
# -----------------------------------------------------------------------------

init_IRA_40 = copy(init_IRA_10)

init_IRA_40['LivPrb'] = 40*[1]
init_IRA_40['T_age'] = 41
init_IRA_40['T_cycle'] = 40
init_IRA_40['PermGroFac'] = 40*[1.01]
init_IRA_40['PermShkStd'] = 35*[0.15] + 5*[0]
init_IRA_40['TranShkStd'] = 35*[0.2] + 5*[0]
init_IRA_40['T_retire'] = 35
init_IRA_40['T_ira'] = 30


# -----------------------------------------------------------------------------
# --- Define all of the parameters for the 30 period consumption IRA model, ---
# --- simplified version to check against the standard ConsInd model. ---------
# -----------------------------------------------------------------------------

init_IRA_30_simp = copy(init_IRA_30)

init_IRA_30_simp['bXtraCount'] = 0

# -----------------------------------------------------------------------------
# --- Define all of the parameters for the 30 period consumption IRA model, ---
# --- simplified version to check against the standard ConsInd model. ---------
# --- This one turns off parallel processing. ---------------------------------
# -----------------------------------------------------------------------------

init_IRA_30_simp_noMP = copy(init_IRA_30_simp)

init_IRA_30_simp_noMP['ParallelBool'] = False # Turn off parallel

# -----------------------------------------------------------------------------
# --- Define all of the parameters for the 30 period consumption IRA model, ---
# --- simplified version to check against the standard ConsInd model. ---------
# --- Also collapses to ConsInd, but allows for two assets --------------------
# -----------------------------------------------------------------------------

init_IRA_30_comp = copy(init_IRA_30)

init_IRA_30_comp['Rsave'] = 1.0 # Never save in this account
init_IRA_30_comp['Rira'] = 1.02 # Make return on IRA same as liquid account
init_IRA_30_comp['T_ira'] = -1 # Turn off IRA penalty
init_IRA_30_comp['MaxIRA'] = 100 # Effectively turn off IRA cap

# -----------------------------------------------------------------------------
# --- Define all of the parameters for the 30 period consumption IRA model, ---
# --- simplified version to check against the standard ConsInd model. ---------
# --- Also collapses to ConsInd, but allows for two assets. -------------------
# --- This version turns off parallel computing. ------------------------------
# -----------------------------------------------------------------------------

init_IRA_30_comp_noMP = copy(init_IRA_30_comp)

init_IRA_30_comp_noMP['ParallelBool'] = False # Turn off parallel

# -----------------------------------------------------------------------------
# --- Define all of the parameters for the 30 period consumption IRA model, ---
# --- full version, without any uncertainty. ----------------------------------
# -----------------------------------------------------------------------------

init_IRA_30_no_shocks = copy(init_IRA_30)
init_IRA_30_no_shocks['AgentCount'] = 1 # no need for multiple simulations
init_IRA_30_no_shocks['aNrmInitStd'] = 0.0 # no wealth hetergeneity
init_IRA_30_no_shocks['PermShkCount'] = 1
init_IRA_30_no_shocks['TranShkCount'] = 1
init_IRA_30_no_shocks['PermShkStd'] = 30*[0.0] # no income shocks
init_IRA_30_no_shocks['TranShkStd'] = 30*[0.0] # no income shocks
init_IRA_30_no_shocks['UnempPrb'] = 0
init_IRA_30_no_shocks['UnempPrbRet'] = 0

# -----------------------------------------------------------------------------
# --- Define all of the parameters for the a one-period IRA model for ---------
# --- testing purposes --------------------------------------------------------
# -----------------------------------------------------------------------------

def cFunc_terminal(m,n):
    return m + n

def dFunc_terminal(m,n):
    return -n

def policyFunc_terminal(m,n):
    return m + n, -n

def vFunc_terminal(m,n):
    return utility(cFunc_terminal(m,n),2)

def vPfunc_terminal(m,n):
    return utilityP(cFunc_terminal(m,n),2)

cFunc = cFunc_terminal
dFunc = cFunc_terminal
vFunc = vFunc_terminal
vPfunc = vPfunc_terminal

solution_terminal = ConsIRASolution(cFunc=cFunc,dFunc=dFunc,vFunc=vFunc,
                                    policyFunc = policyFunc_terminal,
                                    vPfunc=vPfunc,mNrmMin=0.0,hNrm=0.0,
                                    MPCmin=1,MPCmax=1)

IncomeDstn = np.array([[.25,.25,.25,.25],[.9,.9,1.1,1.1],[.8,1.2,.8,1.2]])

init_IRA_Solver = { 'solution_next' : solution_terminal,
                    'IncomeDstn' : IncomeDstn,
                    'LivPrb' : 0.98,
                    'DiscFac' : 0.98,
                    'CRRA' : 2,
                    'Rboro' : 1.2,
                    'Rsave': 1.02,
                    'Rira' : 1.1,
                    'PenIRA' : .1,
                    'MaxIRA' : .2,
                    'DistIRA' : 1,
                    'PermGroFac' : 1.1,
                    'BoroCnstArt' : None,
                    'aXtraGrid' : np.array([.1,.5,1,2,3,4]),
                    'bXtraGrid' : np.array([.5,1,2,3]),
                    'lXtraGrid' : np.array([.5,.75,1,2,3]),
                    'vFuncBool' : True,
                    'CubicBool' : True,
                    'ParallelBool' : True
                    }