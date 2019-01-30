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
** User must add ${HARK_IRA} global to profile.do
** HARK_IRA is the folder where output from HARK is stored

global dataFolder "${HARK_IRA}"	      /*where data files are saved*/
global figureFolder "${HARK_IRA}" /*where graphs are saved*/

cap mkdir ${dataFolder}
cap mkdir ${figureFolder}

set maxvar 15000
tempfile hark

/** Import each HARK csv **/

foreach var in c d m n p t {

	import delimited using /// 
	"${dataFolder}`var'_30.csv" ///
	, clear
	
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

save "${dataFolder}/wide.dta", replace

reshape long c d m n p t, i(id) j(period)

/** Break up dynasties **/

quietly sum id

local nn = r(max)

gen id2 = id + (ceil(period/31) - 1)*r(max)

gen withdraw = (d<0)*d + (d>=0)*0

gen deposit = (d>0)*d + (d<=0)*0
	
save "${dataFolder}/long.dta", replace

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
		
graph export "${figureFolder}/con_liq_ill.png", replace
		
twoway connected withdraw_pos t, ytitle("Withdrawals") /// 
		xtitle("Age (Death at 30)") xline(22)

graph export "${figureFolder}/withdrawal.png", replace
		
twoway connected deposit t, ytitle("Deposits") /// 
		xtitle("Age (Death at 30)") xline(22)
		
graph export "${figureFolder}/deposit.png", replace
		
twoway connected d t, ytitle("Withdrawal + Deposits") /// 
		xtitle("Age (Death at 30)") xline(22)

graph export "${figureFolder}/with_dep.png", replace
