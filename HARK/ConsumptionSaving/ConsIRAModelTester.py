'''
Specifies examples of the full set of parameters required to solve the 
ConsIRAModel.
'''
from __future__ import division, print_function
from copy import copy
import numpy as np

def cFunc_terminal(m,n):
    return m + n

def dFunc_terminal(m,n):
    return -n

def vFunc_terminal(m,n):
    return utility(cFunc_terminal(m,n),2)

def vPfunc_terminal(m,n):
    return utilityP(cFunc_terminal(m,n),2)

cFunc = cFunc_terminal
dFunc = cFunc_terminal
vFunc = vFunc_terminal
vPfunc = vPfunc_terminal

solution_terminal = ConsIRASolution(cFunc=cFunc,dFunc=dFunc,vFunc=vFunc,
                                vPfunc=vPfunc,mNrmMin=0.0,mNrmMin0=0.0,
                                nNrmMin=0.0,hNrm=0.0,MPCmin=1,MPCmax=1)

IncomeDstn = np.array([[.25,.25,.25,.25],[.9,.9,1.1,1.1],[.8,1.2,.8,1.2]])

#agrid = HARKobject
#agrid.aXtraMin = 0.001
#agrid.aXtraMax = 20
#agrid.aXtraCount = 48
#agrid.aXtraNestFac = 3
#agrid.aXtraExtra = np.array([None])
#aXtraGrid = constructAssetsGrid(agrid)
#lXtraGrid = constructAssetsGrid(agrid)
#bXtraGrid = constructAssetsGrid(agrid)

ira_params = { 'solution_next' : solution_terminal,
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
                #'aXtraGrid' : aXtraGrid,
                #'bXtraGrid' : bXtraGrid,
                #'lXtraGrid' : lXtraGrid,
                'vFuncBool' : True,
                'CubicBool' : True
                }

ex = ConsIRASolver(**ira_params)
ex.prepareToSolve()
ex.prepareToCalcEndOfPrdvAndvP()
EndOfPrdv,EndOfPrdvP = ex.calcEndOfPrdvAndvP()
cNrm,lNrm,bNrm = ex.getPointsForPureConsumptionInterpolation(EndOfPrdv,EndOfPrdvP,ex.aNrmNow,ex.bNrmNow)
ex.makePurecFunc(cNrm,lNrm,bNrm)
ex.makeEndOfPrdvFunc(EndOfPrdv)
ex.makecAnddFunc()
