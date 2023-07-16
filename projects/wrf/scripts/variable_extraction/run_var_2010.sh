#!/bin/bash
#----------
# run year
folder=/gpfs/data1/lobodagp/hej/chp3/Lightning
year=2010
#----------
dates=${folder}/${year}/${year}*
for date in ${dates}
do
    echo ${date} 
    datestr=${date##*/}
    month=${datestr:4:2}
    day=${datestr:6:2}
    echo ${month}
    echo ${day}
    #-------------------------------------------------------
    #   PREPARE OUTPUT DIRECTORIES
    #-------------------------------------------------------
    mkdir ${date}/Cloud_vars
    mkdir ${date}/Lightning_wrf
    mkdir ${date}/Stability_indices
    mkdir ${date}/Weather_vars
    #-------------------------------------------------------
    #   RUN NCL SCRIPTS
    #-------------------------------------------------------
    echo "1.upp_cloud"
    ncl 'year_argv="'${year}'"' 'date_argv="'${datestr}'"' upp_cloud.ncl
    #--------------------
    echo "2.wrf_cloud"
    ncl 'year_argv="'${year}'"' 'date_argv="'${datestr}'"' wrf_cloud.ncl
    #--------------------
    echo "3.wrf_indices"
    ncl 'year_argv="'${year}'"' 'date_argv="'${datestr}'"' wrf_indices.ncl
    #--------------------
    echo "4.wrf_lightning"
    ncl 'year_argv="'${year}'"' 'date_argv="'${datestr}'"' wrf_lightning.ncl
    #--------------------
    echo "5.wrf_weather"
    ncl 'year_argv="'${year}'"' 'date_argv="'${datestr}'"' wrf_weather.ncl
    #--------------------
    echo "6.wrf_wp"
    ncl 'year_argv="'${year}'"' 'date_argv="'${datestr}'"' wrf_wp.ncl
done
