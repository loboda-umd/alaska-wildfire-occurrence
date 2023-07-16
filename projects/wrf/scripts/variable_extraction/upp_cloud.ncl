
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

  ; ---------------------
  ntimes = 9
  do time = 0,ntimes-1
    hrs = sprinti("%0.2i", time*3)
    ;-----------------------------------------------------------------------
    ; read data
    a = addfile(str_concat((/dir_name,"postprd/WRFPRS_d02.",hrs,".grb"/)),"r")

    ; --------------------------------------------------
    fout_strs = (/dir_name,"Cloud_vars/UPP_",date,"_",tostring(time*3),"hr.nc"/)
    fout_name = str_concat(fout_strs)
    system(str_concat((/"rm -f ",fout_name/)))
    fout = addfile(fout_name,"c")

    ;-----------------------------------------------------------------------
    ; Extract variables
    LI = a->4LFTX_GDS3_SPDY
    PLI = a->PLI_GDS3_SPDY
    Haines = a->HINDEX_GDS3_SFC
    CF_total = a->T_CDC_GDS3_EATM
    CF_conv = a->CDCON_GDS3_EATM
    CF_low = a->L_CDC_GDS3_LCY
    CF_mid = a->M_CDC_GDS3_MCY
    CF_high = a->H_CDC_GDS3_HCY
    CW_total = a->TCOLW_GDS3_EATM
    CI_total = a->TCOLI_GDS3_EATM
    CR_total = a->TCOLR_GDS3_EATM
    CS_total = a->TCOLS_GDS3_EATM
    ; -----------------------------------
    BT = a->BRTMP_GDS3_NTAT
    CTT = a->TMP_GDS3_CTL
    CTP = a->PRES_GDS3_CTL
    CTH = a->HGT_GDS3_CTL

    ; -----------------------------------------------------------------------
    ; SAVE TO OUTPUT FILE
    fout->LI = LI
    fout->PLI = PLI
    fout->Haines = Haines
    fout->CF_total = CF_total
    fout->CF_conv = CF_conv
    fout->CF_low = CF_low
    fout->CF_mid = CF_mid
    fout->CF_high = CF_high
    fout->CW_total = CW_total
    fout->CI_total = CI_total
    fout->CR_total = CR_total
    fout->CS_total = CS_total
    fout->BT = BT
    fout->CTT = CTT
    fout->CTP = CTP
    fout->CTH = CTH

  end do
   
end
