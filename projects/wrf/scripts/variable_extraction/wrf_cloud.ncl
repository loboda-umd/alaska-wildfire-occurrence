
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
  ; dir_name = "/gpfs/data1/lobodagp/hej/WRF_cases_modeling/size_25km_5km/2007/"
  ; date = "2007-07-11"
  
  dir_name = str_concat((/"/gpfs/data1/lobodagp/hej/chp3/Lightning/",year_argv,"/",date_argv,"/"/))
  ; -----------
  year = str_get_cols(date_argv, 0, 3)
  month = str_get_cols(date_argv, 4, 5)
  day = str_get_cols(date_argv, 6, 7)
  date = str_join((/year, month, day/),"-")

  ;-----------------------------------------------------------------------
  ; read data
  a = addfile(str_concat((/dir_name,"wrfout_d02_",date,"_00:00:00.nc"/)),"r")
  aux = addfile(str_concat((/dir_name,"auxhist24_d02_",date,"_00:00:00.nc"/)),"r")

  times = wrf_user_getvar(a,"times",-1)  ; get all times in the file
  ntimes = dimsizes(times)         ; number of times in the file
  print(ntimes)

  do time = 0,ntimes-1
    ; print(time)
    ; --------------------------------------------------
    fout_strs = (/dir_name,"Cloud_vars/WRFaux_",date,"_",tostring(time*3),"hr.nc"/)
    fout_name = str_concat(fout_strs)
    system(str_concat((/"rm -f ",fout_name/)))
    fout = addfile(fout_name,"c")

    ; --------------------------------------------------
    ; Get LAT & LON & p
    lat = wrf_user_getvar(a,"lat",time)
    lon = wrf_user_getvar(a,"lon",time)
    
    ; ------------------------------------------------
    ; CLOUD EFFECTIVE RADIUS
    re_cloud = aux->RE_CLOUD(time,:,:,:)
    ; print(dimsizes(re_cloud))
    CER_max = dim_max_n(re_cloud, 0)*100000
    CER_avg = dim_avg_n(re_cloud, 0)*100000
    CER_min = dim_min_n(re_cloud, 0)*100000
    ; ICE EFFECTIVE RADIUS
    re_ice = aux->RE_ICE(time,:,:,:)
    IER_max = dim_max_n(re_ice, 0)*100000
    IER_avg = dim_avg_n(re_ice, 0)*100000
    IER_min = dim_min_n(re_ice, 0)*100000

    ; ------------------------------------------------
    ; MIXING RATIO / NUMBER CONCENTRATION
    ; --- cloud
    qcloud = a->QCLOUD(time,:,:,:)    
    QC_max = dim_max_n(qcloud, 0)*100000
    QC_avg = dim_avg_n(qcloud, 0)*100000
    QC_min = dim_min_n(qcloud, 0)*100000
    ; --- ice
    qice = a->QICE(time,:,:,:)
    QI_max = dim_max_n(qice, 0)*100000
    QI_avg = dim_avg_n(qice, 0)*100000
    QI_min = dim_min_n(qice, 0)*100000
    qnice = a->QNICE(time,:,:,:)
    QNI_max = dim_max_n(qnice, 0)
    QNI_avg = dim_avg_n(qnice, 0)
    QNI_min = dim_min_n(qnice, 0)
    ; --- rain
    qrain = a->QRAIN(time,:,:,:)
    QR_max = dim_max_n(qrain, 0)*100000
    QR_avg = dim_avg_n(qrain, 0)*100000
    QR_min = dim_min_n(qrain, 0)*100000
    qnrain = a->QNRAIN(time,:,:,:)
    QNR_max = dim_max_n(qnrain, 0)
    QNR_avg = dim_avg_n(qnrain, 0)
    QNR_min = dim_min_n(qnrain, 0)

    ;-----------------------------------------------------------------------
    ; Save data
    fout->lat = lat
    fout->lon = lon
    ;---------------
    fout->CER_max = CER_max
    fout->CER_avg = CER_avg
    fout->CER_min = CER_min
    fout->IER_max = IER_max
    fout->IER_avg = IER_avg
    fout->IER_min = IER_min
    ;----------------
    fout->QC_max = QC_max
    fout->QC_avg = QC_avg
    fout->QC_min = QC_min
    ;----------------
    fout->QI_max = QI_max
    fout->QI_avg = QI_avg
    fout->QI_min = QI_min
    fout->QNI_max = QNI_max
    fout->QNI_avg = QNI_avg
    fout->QNI_min = QNI_min
    ;----------------
    fout->QR_max = QR_max
    fout->QR_avg = QR_avg
    fout->QR_min = QR_min
    fout->QNR_max = QNR_max
    fout->QNR_avg = QNR_avg
    fout->QNR_min = QNR_min

  end do 
   
end
