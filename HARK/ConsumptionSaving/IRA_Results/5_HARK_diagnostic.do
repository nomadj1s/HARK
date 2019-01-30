********************************************************************
**  TITLE:		4_HARK_simulation.do
**	PURPOSE: 	Take output from HARK simulation and plot outcomes
**
**	INPUTS:		.csv files from HARK
**	
**	OUTPUTS:	
**
**	NOTES:		User must set local file and folder locations.
**
********************************************************************

clear all
set more off
set scheme s1color
cap log close

** Set Folders
** User must add ${HARK_IRA} global to profile.do
** HARK_IRA is the folder where output from HARK is stored

global dataFolder "${HARK_IRA}"	      /*where data files are saved*/
global figureFolder "${HARK_IRA}" /*where graphs are saved*/

cap mkdir ${dataFolder}
cap mkdir ${figureFolder}

import delimited using "${HARK_IRA}/IRA_Kinked_data_simple.csv", clear

sort period mrange

forval x = 15(5)25 {

	twoway line ckinked cira mrange if period == `x' & mrange < 2 & /// 
	mrange > 0, legend(label(1 "ConsInd Kinked") label(2 "NEGM Model")) /// 
	ytitle("Consumption") xtitle("Liquid Assets") title("Period `x'/30") ///
	name(p`x', replace)
	
	graph export "${HARK_IRA}/HARK_Output/check_p`x'.png", replace
	
}
