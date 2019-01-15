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
**	Last Update:	11/05/2018
********************************************************************

clear all
set more off
set scheme s1color
cap log close

** Set Folders
** User must add ${iraWithdrawals} global to profile.do

global logFolder "${iraWithdrawals}/code/logs"	   /*where Log files are saved*/
global dataFolder "${iraWithdrawals}/data"	      /*where data files are saved*/
global figureFolder "${iraWithdrawals}/results/figures" /*where 
                                                              graphs are saved*/
global tableFolder "${iraWithdrawals}/results/tables" /*where talbes are saved*/
global codeFolder "${iraWithdrawals}/results/code"  /*where do files are saved*/

cap mkdir ${logFolder}
cap mkdir ${dataFolder}
cap mkdir ${figureFolder}
cap mkdir ${tableFolder}
cap mkdir ${codeFolder}

set maxvar 15000
tempfile hark

/** Import each HARK csv **/

foreach var in c d m n p t {

	import delimited using /// 
	"${dataFolder}/HARK_Output/`var'.csv" ///
	, clear
	
	xpose, clear
	
	gen id = _n
	
	forval x = 1/120 {
		rename v`x' `var'`x'
	}

	if "`var'" == "c" {
		save `hark', replace
	}
	else {
		merge 1:1 id using `hark'
		drop _merge
		save `hark', replace
	}

}

save "${iraWithdrawals}/data/HARK_Output/wide.dta", replace

reshape long c d m n p t, i(id) j(period)

/** Break up dynasties **/

quietly sum id

local nn = r(max)

gen id2 = id + (ceil(period/31) - 1)*r(max)

gen withdraw = (d<0)*d + (d>=0)*0

gen deposit = (d>0)*d + (d<=0)*0
	
save "${iraWithdrawals}/data/HARK_Output/long.dta", replace

/** Create Graphs **/

collapse n m d c p withdraw deposit, by(t)

drop if t == 31

foreach x in c d m n withdraw deposit {
	replace `x' = `x'*p
}

gen withdraw_pos = -withdraw

twoway connected c m n t, legend(label(1 "Consumption") /// 
		label(2 "Liquid Assets") label(3 "Illiquid Assets")) xline(22) ///
		ytitle("Amount") xtitle("Age (Death at 30)")
		
graph export "${figureFolder}/Hark_Output/con_liq_ill.png", replace
		
twoway connected withdraw_pos t, ytitle("Withdrawals") /// 
		xtitle("Age (Death at 30)") xline(22)

graph export "${figureFolder}/Hark_Output/withdrawal.png", replace
		
twoway connected deposit t, ytitle("Deposits") /// 
		xtitle("Age (Death at 30)") xline(22)
		
graph export "${figureFolder}/Hark_Output/deposit.png", replace
		
twoway connected d t, ytitle("Withdrawal + Deposits") /// 
		xtitle("Age (Death at 30)") xline(22)

graph export "${figureFolder}/Hark_Output/with_dep.png", replace
