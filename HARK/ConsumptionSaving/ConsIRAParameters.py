'''
Specifies examples of the full set of parameters required to solve the 
ConsIRAModel.
'''
from __future__ import division, print_function
from copy import copy
import numpy as np

# -----------------------------------------------------------------------------
# --- Define all of the parameters for the consumption IRA model   ------------
# -----------------------------------------------------------------------------

CRRA = 2.0                          # Coefficient of relative risk aversion
DiscFac = 0.96                      # Intertemporal discount factor
                                    # Survival probability
LivPrb = [0.99,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]

AgentCount = 10000                  # Number of agents of this type (only matters for simulation)
aNrmInitMean = 0.0                  # Mean of log initial assets (only matters for simulation)
aNrmInitStd  = 1.0                  # Standard deviation of log initial assets (only for simulation)
pLvlInitMean = 0.0                  # Mean of log initial permanent income (only matters for simulation)
pLvlInitStd  = 0.0                  # Standard deviation of log initial permanent income (only matters for simulation)
PermGroFacAgg = 1.0                 # Aggregate permanent income growth factor (only matters for simulation)
T_age = 11                          # Age after which simulated agents are automatically killed
T_cycle = 10                        # Number of periods in the cycle for this agent type

# Parameters for constructing the "assets above minimum" grid
aXtraMin = 0.001                    # Minimum end-of-period "assets above minimum" value
aXtraMax = 20                       # Maximum end-of-period "assets above minimum" value
aXtraExtra = None                   # Some other value of "assets above minimum" to add to the grid, not used
aXtraNestFac = 3                    # Exponential nesting factor when constructing "assets above minimum" grid
aXtraCount = 48                     # Number of points in the grid of "assets above minimum"

bXtraMin = 0.001                    # Minimum end-of-period "assets above minimum" value
bXtraMax = 20                       # Maximum end-of-period "assets above minimum" value
bXtraExtra = None                   # Some other value of "assets above minimum" to add to the grid, not used
bXtraNestFac = 3                    # Exponential nesting factor when constructing "assets above minimum" grid
bXtraCount = 48                     # Number of points in the grid of "assets above minimum"

# Parameters describing the income process
PermShkCount = 7                    # Number of points in discrete approximation to permanent income shocks
TranShkCount = 7                    # Number of points in discrete approximation to transitory income shocks
                                    # Standard deviation of log permanent income shocks
                                    # Permanent income growth factor
PermGroFac = [1.01,1.01,1.01,1.01,1.01,1.02,1.02,1.02,1.02,1.02]
PermShkStd = [0.1,0.2,0.1,0.2,0.1,0.2,0.1,0,0,0]                  
                                    # Standard deviation of log transitory income shocks
TranShkStd = [0.3,0.2,0.1,0.3,0.2,0.1,0.3,0,0,0]
UnempPrb = 0.05                     # Probability of unemployment while working
UnempPrbRet = 0.005                 # Probability of "unemployment" while retired
IncUnemp = 0.3                      # Unemployment benefits replacement rate
IncUnempRet = 0.0                   # "Unemployment" benefits when retired
tax_rate = 0.0                      # Flat income tax rate
T_retire = 7                        # Period of retirement (0 --> no retirement)

# A few other parameters
BoroCnstArt = 0.0                   # Artificial borrowing constraint; imposed minimum level of end-of period assets
CubicBool = False                   # Use cubic spline interpolation when True, linear interpolation when False
vFuncBool = True                    # Whether to calculate the value function during solution

# Interest rate/factors

Rboro = 1.20                        # Interest factor on assets when borrowing, a < 0
Rsave = 1.02                        # Interest factor on assets when saving, a > 0
Rira = 1.10                         # Interest factor on illiquid (IRA) assets

# Make a dictionary to specify an IRA consumer type
init_IRA = { 'CRRA': CRRA,
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
             'T_retire':T_retire,
             'aNrmInitMean' : aNrmInitMean,
             'aNrmInitStd' : aNrmInitStd,
             'pLvlInitMean' : pLvlInitMean,
             'pLvlInitStd' : pLvlInitStd,
             'PermGroFacAgg' : PermGroFacAgg,
             'T_age' : T_age,
             'T_cycle' : T_cycle
            }
