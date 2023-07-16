#!/bin/bash
#--------------
folder=/gpfs/data1/lobodagp/hej/chp3/Lightning
for year in 2002 2006 2007 2008 2010 2013 2015 2017
do 
	dates=${folder}/${year}/${year}*
	#---
	for date in ${dates}
	do
		echo ${date} 
		datestr=${date##*/}
		month=${datestr:4:2}
		day=${datestr:6:2}
		echo ${month}
		echo ${day}
		mkdir ${date}/postprd
		mkdir ${date}/parm
		mkdir ${date}/wrfprd
		cp /gpfs/data1/lobodagp/hej/chp3/WRF_simulations/${year}/wrfout/auxhist24_d02_${year}-${month}-${day}* ${date}/
		mv ${date}/wrfprd/wrfout* ${date}/
	done
done
