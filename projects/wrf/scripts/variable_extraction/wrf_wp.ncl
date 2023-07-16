
;   Example script - read and plot data from WRF time series output
;   November 2008

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "/gpfs/data1/lobodagp/hej/NCL/examples/writefiles/create_netcdf_file_utils.ncl"

begin

  ; ----------------------------------------------------------------
  ; MODIFY INPUT FILE SETTINGS
  ; dir_name = "/gpfs/data1/lobodagp/hej/WRF_cases_modeling/size_25km_5km/2015/"
  ; date = "2015-07-14"
  dir_name = str_concat((/"/gpfs/data1/lobodagp/hej/chp3/Lightning/",year_argv,"/",date_argv,"/"/))
  ; -----------
  year = str_get_cols(date_argv, 0, 3)
  month = str_get_cols(date_argv, 4, 5)
  day = str_get_cols(date_argv, 6, 7)
  date = str_join((/year, month, day/),"-")


  ;-----------------------------------------------------------------------
  ; read data
  a = addfile(str_concat((/dir_name,"wrfout_d02_",date,"_00:00:00.nc"/)),"r")
  cf = addfile(str_concat((/dir_name,"wrfpost_cf_",date,".nc"/)),"r")

  ;-----------------------------------------------------------------------
  ; extract variables

  times = wrf_user_getvar(a,"times",-1)  ; get all times in the file
  ntimes = dimsizes(times)         ; number of times in the file
  print(ntimes)

  do time = 0,ntimes-1
    ; print(time)

    ; --------------------------------------------------
    fout_strs = (/dir_name,"Cloud_vars/Waterpath_",date,"_",tostring(time*3),"hr.nc"/)
    fout_name = str_concat(fout_strs)
    system(str_concat((/"rm -f ",fout_name/)))
    fout = addfile(fout_name,"c")

    ; --------------------------------------------------
    ; Get LAT & LON & p
    lat = wrf_user_getvar(a,"lat",time)
    lon = wrf_user_getvar(a,"lon",time)
    

    ; ------------------------------------------
    ; Liquid water path
    LWP = cf->LWP(time,:,:)
    IWP = cf->IWP(time,:,:)
    ;-----------------------------------------------------------------------
    ; Save data
    fout->lat = lat
    fout->lon = lon
    ;---------------
    fout->LWP = LWP
    fout->IWP = IWP

  end do 
   
end
