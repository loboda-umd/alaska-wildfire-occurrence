
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

  ; dir_name = "/gpfs/data1/lobodagp/hej/WRF_cases_modeling/2007/1day/"
  ; date = "2007-07-11"


  do time = 0,ntimes-1
    ; res@TimeLabel = times(time)

    ; --------------------------------------------------
    fout_strs = (/dir_name,"Stability_indices/SI_",date,"_",tostring(time*3),"hr.nc"/)
    fout_name = str_concat(fout_strs)
    system(str_concat((/"rm -f ",fout_name/)))
    fout = addfile(fout_name,"c")

    ; --------------------------------------------------
    ; Get LAT & LON
    lat = wrf_user_getvar(a,"lat",time)
    lon = wrf_user_getvar(a,"lon",time)

    ; ; Get PW
    ; pw = wrf_user_getvar(a,"pw",time)

    ; Get 3d variables
    td = wrf_user_getvar(a,"td",time)           ; Dewpoint temperature
    tc = wrf_user_getvar(a,"tc",time)           ; Temperature [C]
    p = wrf_user_getvar(a,"pressure",time)      ; Total pressure
    z = wrf_user_getvar(a,"height",time)        ; Full model height (m)
    
    ; Get dewpoint temperature at certain pressure levels
    td_500 = wrf_user_interp_level(td, p, 500, False)
    td_700 = wrf_user_interp_level(td, p, 700, False)
    td_850 = wrf_user_interp_level(td, p, 850, False)

    ; Get temperature [C] at certain pressure levels
    tc_500 = wrf_user_interp_level(tc, p, 500, False)
    tc_700 = wrf_user_interp_level(tc, p, 700, False)
    tc_850 = wrf_user_interp_level(tc, p, 850, False)

    ; Get geopotential height at certain pressure levels
    z_700 = wrf_user_interp_level(z, p, 700, False)
    z_1000 = wrf_user_interp_level(z, p, 1000, False)

    ;-----------------------------------------------------------------------
    ; Calculate indices
    ; -> K index
    KI = (tc_850 - tc_500) + td_850 - (tc_700 - td_700)

    ; -> Total totals
    TT = tc_850 + td_850 - 2*tc_500

    ; -> Showalter index (SHOW)
    tp_850 = (tc_850 + 273.15)*((500/850)^0.286) - 273.15
    SHOW = tc_500 - tp_850

    ; -> Boyden index (BI)
    BI = (z_700 - z_1000)/10 - tc_700 - 200

    ;-----------------------------------------------------------------------
    ; Extract existing indices with NCL
    cape2d = wrf_user_getvar(a,"cape_2d",time) 

    CAPE = cape2d(0,:,:)
    CIN = cape2d(1,:,:)
    LCL = cape2d(2,:,:)
    LFC = cape2d(3,:,:)


    ;-----------------------------------------------------------------------
    ; LIGHTING PARAMETERIZATION
    ; LPI

    ; PR92


    ;-----------------------------------------------------------------------
    ; Save data
    fout->lat = lat
    fout->lon = lon
    fout->CAPE = CAPE
    fout->CIN = CIN
    fout->LCL = LCL
    fout->LFC = LFC 
    fout->BI = BI
    fout->KI = KI
    fout->TT = TT
    fout->SHOW = SHOW

  end do 
   
end
