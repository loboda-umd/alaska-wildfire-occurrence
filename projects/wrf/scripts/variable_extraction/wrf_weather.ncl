
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

  ;-----------------------------------------------------------------------
  ; extract variables

  times = wrf_user_getvar(a,"times",-1)  ; get all times in the file
  ntimes = dimsizes(times)         ; number of times in the file
  print(ntimes)

  do time = 0,ntimes-1
    ; print(time)

    ; --------------------------------------------------
    fout_strs = (/dir_name,"Weather_vars/Weather_",date,"_",tostring(time*3),"hr.nc"/)
    fout_name = str_concat(fout_strs)
    system(str_concat((/"rm -f ",fout_name/)))
    fout = addfile(fout_name,"c")

    ; --------------------------------------------------
    ; Get LAT & LON & p
    lat = wrf_user_getvar(a,"lat",time)
    lon = wrf_user_getvar(a,"lon",time)
    p = wrf_user_getvar(a,"pressure",time)      ; Total pressure
    
    ; ------------------------------------------------
    ; Get PW -> 2d variable accumulated precipitable water
    pw = wrf_user_getvar(a,"pw",time)

    ; -------------------------------------------------
    ; precipication
    rain_exp = wrf_user_getvar(a,"RAINNC",time)
    rain_con = wrf_user_getvar(a,"RAINC",time)
    rain_tot = rain_exp + rain_con

    ; ------------------------------------------------
    ; Get Air Temperature (C)
    t2 = a->T2(time,:,:)
    t2 = t2 - 273.15
    ; Get temperature [C] at certain pressure levels
    tc = wrf_user_getvar(a,"tc",time)           ; Temperature [C]
    tc_500 = wrf_user_interp_level(tc, p, 500, False)
    tc_700 = wrf_user_interp_level(tc, p, 700, False)
    tc_850 = wrf_user_interp_level(tc, p, 850, False)
    tc_1000 = wrf_user_interp_level(tc, p, 1000, False)

    ; --------------------------------------------------
    ; Get Dewpoint Temperature (C)
    td2 = wrf_user_getvar(a,"td2",time)  
    ; Get dewpoint temperature at certain pressure levels
    td = wrf_user_getvar(a,"td",time)           ; Dewpoint temperature    
    td_500 = wrf_user_interp_level(td, p, 500, False)
    td_700 = wrf_user_interp_level(td, p, 700, False)
    td_850 = wrf_user_interp_level(td, p, 850, False)
    td_1000 = wrf_user_interp_level(td, p, 1000, False)
    
    ; ---------------------------------------------------
    ; Get Relative Humidity (%)
    rh2 = wrf_user_getvar(a,"rh2",time)
    ; Get rh at certain pressure levels
    rh = wrf_user_getvar(a,"rh",time)           ; Relative Humidity 
    rh_500 = wrf_user_interp_level(rh, p, 500, False)
    rh_700 = wrf_user_interp_level(rh, p, 700, False)
    rh_850 = wrf_user_interp_level(rh, p, 850, False)
    rh_1000 = wrf_user_interp_level(rh, p, 1000, False)
 
    ; ----------------------------------------------------
    ; Sea Level Pressure (hPa)
    slp = wrf_user_getvar(a,"slp",time)

    ; ---------------------------------------------------
    ; Geopotential Height
    geopt = wrf_user_getvar(a, "geopotential",time)
    gpz = geopt/9.81
    gpz_500 = wrf_user_interp_level(gpz, p, 500, False)
    gpz_700 = wrf_user_interp_level(gpz, p, 700, False)
    gpz_850 = wrf_user_interp_level(gpz, p, 850, False)
    gpz_1000 = wrf_user_interp_level(gpz, p, 1000, False)
    ; ------------------------
    ; layer thickness
    dz1 = gpz_500 - gpz_1000
    dz2 = gpz_850 - gpz_1000
    dz3 = gpz_700 - gpz_850
    dz4 = gpz_700 - gpz_1000

    ; ---------------------------------------------------
    ; helicity
    helicity = wrf_user_getvar(a,"helicity",time)
    uhelicity = wrf_user_getvar(a,"updraft_helicity",time)

    ; ----------------------------------------------------
    ; vertical wind speed
    wa = wrf_user_getvar(a,"wa",time)
    wa_500 = wrf_user_interp_level(wa, p, 500, False)
    wa_700 = wrf_user_interp_level(wa, p, 700, False)
    wa_850 = wrf_user_interp_level(wa, p, 850, False)
    wa_1000 = wrf_user_interp_level(wa, p, 1000, False)


    ;-----------------------------------------------------------------------
    ; Save data
    fout->lat = lat
    fout->lon = lon
    ;---------------
    fout->pw = pw
    fout->rain_tot = rain_tot
    fout->slp = slp
    fout->helicity = helicity
    fout->uhelicity = uhelicity
    ;---------------
    fout->t2 = t2
    fout->tc_500 = tc_500
    fout->tc_750 = tc_700
    fout->tc_800 = tc_850
    fout->tc_1000 = tc_1000
    ;---------------
    fout->td2 = td2
    fout->td_500 = td_500
    fout->td_750 = td_700
    fout->td_800 = td_850
    fout->td_1000 = td_1000
    ;---------------
    fout->rh2 = rh2
    fout->rh_500 = rh_500
    fout->rh_750 = rh_700
    fout->rh_800 = rh_850
    fout->rh_1000 = rh_1000
    ;---------------
    fout->gpz_500 = gpz_500
    fout->gpz_750 = gpz_700
    fout->gpz_800 = gpz_850
    fout->gpz_1000 = gpz_1000
    ;--------------
    fout->dz1 = dz1
    fout->dz2 = dz2
    fout->dz3 = dz3
    fout->dz4 = dz4
    ;---------------
    fout->wa_500 = wa_500
    fout->wa_750 = wa_700
    fout->wa_800 = wa_850
    fout->wa_1000 = wa_1000

  end do 
   
end
