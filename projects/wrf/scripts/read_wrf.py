from __future__ import print_function

from netCDF4 import Dataset
from wrf import getvar, interplevel

# get WRF output filename and place it into a netcdf dataset
filename = '/explore/nobackup/projects/ilab/projects/LobodaTFO/software/WRF/run/wrfout_d02_2015-07-14_00:00:00'
dset = Dataset(filename)

# CFtotal - how to get total?, CFlow
# problem: CFTotal as the sum is coming as 2.xxx which is higher than 1
cloudfrac = getvar(dset, "cloudfrac")
low_cloudfrac = cloudfrac[0, :, :]
mid_cloudfrac = cloudfrac[1, :, :]
high_cloudfrac = cloudfrac[2, :, :]
total_cloudfrac = low_cloudfrac + mid_cloudfrac + high_cloudfrac
print("Cloudfrac", cloudfrac.shape, total_cloudfrac.shape)

# GPZ
pressure = getvar(dset, "pressure")
gpz = getvar(dset, "geopt") / 9.81
print(pressure.shape, gpz.shape)

# GPZ500, GPZ700, GPZ850
gpz_500 = interplevel(gpz, pressure, 500)
gpz_700 = interplevel(gpz, pressure, 700)
gpz_850 = interplevel(gpz, pressure, 850)
gpz_1000 = interplevel(gpz, pressure, 1000)
print(gpz_500.shape)

# DZ700-850
dz500_1000 = gpz_500 - gpz_1000
dz850_1000 = gpz_850 - gpz_1000
dz700_850 = gpz_700 - gpz_850
dz700_1000 = gpz_700 - gpz_1000

# Helicity
helicity = getvar(dset, "helicity")
print(helicity.shape)

# LCL
lcl = getvar(dset, "lcl")
print(lcl.shape)

# PW
pw = getvar(dset, "pw")
print(pw.shape)

# SLP
slp = getvar(dset, "slp")
print(slp.shape)

# RH2
rh2 = getvar(dset, "rh2")
print("rh2", rh2.shape)

# RH
rh = getvar(dset, "rh")
rh_500 = interplevel(rh, pressure, 500)
rh_700 = interplevel(rh, pressure, 700)
rh_800 = interplevel(rh, pressure, 800)
rh_850 = interplevel(rh, pressure, 850)

# T2
t2 = getvar(dset, "T2")
print("t2", t2.shape)

# Td2
td2 = getvar(dset, "td2")
print("td2", td2.shape)

# Td
td = getvar(dset, "td")
td_500 = interplevel(td, pressure, 500)
td_850 = interplevel(td, pressure, 850)

# TC
tc = getvar(dset, "tc")
tc_500 = interplevel(tc, pressure, 500)
tc_750 = interplevel(tc, pressure, 700) # not sure why 750 goes with 700 interp, reference says so
tc_850 = interplevel(tc, pressure, 850)
tp_850 = (tc_850 + 273.15) * ((500 / 850)**0.286) - 273.15

# SHOW
show = tc_500 - tp_850

# TT
tt = tc_850 + td_850 - 2 * tc_500

# Rain
rain_exp = getvar(dset, "RAINNC")  # not for training
rain_con = getvar(dset, "RAINC")  # not for training
rain_tot = rain_exp + rain_con

# W500
wa = getvar(dset, "wa")
wa_500 = interplevel(wa, pressure, 500)

lpi = getvar(dset, "LPI")
#pr92 = getvar(dset, "CG_FLASHCOUNT")

"""
  BT - missing
  CFtotal - missing
  PLI - missing


(ilab-tensorflow) [jacaraba@gpu007 scripts]$ grep -ri BT variable_extraction/.
variable_extraction/./upp_cloud.ncl:    BT = a->BRTMP_GDS3_NTAT
variable_extraction/./upp_cloud.ncl:    fout->BT = BT
(ilab-tensorflow) [jacaraba@gpu007 scripts]$ grep -ri PLI variable_extraction/.
variable_extraction/./upp_cloud.ncl:    PLI = a->PLI_GDS3_SPDY
variable_extraction/./upp_cloud.ncl:    fout->PLI = PLI
(ilab-tensorflow) [jacaraba@gpu007 scripts]$ grep -ri CF variable_extraction/.
variable_extraction/./upp_cloud.ncl:    CF_total = a->T_CDC_GDS3_EATM
variable_extraction/./upp_cloud.ncl:    CF_conv = a->CDCON_GDS3_EATM
variable_extraction/./upp_cloud.ncl:    CF_low = a->L_CDC_GDS3_LCY
variable_extraction/./upp_cloud.ncl:    CF_mid = a->M_CDC_GDS3_MCY
variable_extraction/./upp_cloud.ncl:    CF_high = a->H_CDC_GDS3_HCY
variable_extraction/./upp_cloud.ncl:    fout->CF_total = CF_total
variable_extraction/./upp_cloud.ncl:    fout->CF_conv = CF_conv
variable_extraction/./upp_cloud.ncl:    fout->CF_low = CF_low
variable_extraction/./upp_cloud.ncl:    fout->CF_mid = CF_mid
variable_extraction/./upp_cloud.ncl:    fout->CF_high = CF_high
variable_extraction/./wrf_wp.ncl:  cf = addfile(str_concat((/dir_name,"wrfpost_cf_",date,".nc"/)),"r")
variable_extraction/./wrf_wp.ncl:    LWP = cf->LWP(time,:,:)
variable_extraction/./wrf_wp.ncl:    IWP = cf->IWP(time,:,:)
"""

"""
Lightning parameterizations
PR92 Flash distribution of CG lightning with PR92 
LPI Lightning probability index
"""

"""
dict_keys(['Times', 'XLAT', 'XLONG', 'LU_INDEX', 'ZNU', 'ZNW', 'ZS', 'DZS', 'VAR_SSO',
'BATHYMETRY_FLAG', 'U', 'V', 'W', 'PH', 'PHB', 'T', 'THM', 'HFX_FORCE', 'LH_FORCE', 'TSK_FORCE',
'HFX_FORCE_TEND', 'LH_FORCE_TEND', 'TSK_FORCE_TEND', 'MU', 'MUB', 'NEST_POS', 'P', 'PB', 'FNM', 'FNP',
'RDNW', 'RDN', 'DNW', 'DN', 'CFN', 'CFN1', 'THIS_IS_AN_IDEAL_RUN', 'P_HYD', 'Q2', 'T2', 'TH2', 'PSFC',
'U10', 'V10', 'LPI', 'RDX', 'RDY', 'AREA2D', 'DX2D', 'RESM', 'ZETATOP', 'CF1', 'CF2', 'CF3', 'ITIMESTEP',
'XTIME', 'QVAPOR', 'QCLOUD', 'QRAIN', 'QICE', 'QSNOW', 'QGRAUP', 'QNICE', 'QNRAIN', 'SHDMAX', 'SHDMIN', 
'SNOALB', 'TSLB', 'SMOIS', 'SH2O', 'SMCREL', 'SEAICE', 'XICEM', 'SFROFF', 'UDROFF', 'IVGTYP', 'ISLTYP', 
'VEGFRA', 'GRDFLX', 'ACGRDFLX', 'ACSNOM', 'SNOW', 'SNOWH', 'CANWAT', 'SSTSK', 'WATER_DEPTH', 'COSZEN', 
'LAI', 'U10E', 'V10E', 'DTAUX3D', 'DTAUY3D', 'DUSFCG', 'DVSFCG', 'VAR', 'CON', 'OA1', 'OA2', 'OA3', 'OA4', 
'OL1', 'OL2', 'OL3', 'OL4', 'TKE_PBL', 'EL_PBL', 'O3_GFS_DU', 'MAPFAC_M', 'MAPFAC_U', 'MAPFAC_V', 'MAPFAC_MX', 
'MAPFAC_MY', 'MAPFAC_UX', 'MAPFAC_UY', 'MAPFAC_VX', 'MF_VX_INV', 'MAPFAC_VY', 'F', 'E', 'SINALPHA', 'COSALPHA', 
'HGT', 'TSK', 'P_TOP', 'GOT_VAR_SSO', 'T00', 'P00', 'TLP', 'TISO', 'TLP_STRAT', 'P_STRAT', 'MAX_MSFTX', 
'MAX_MSFTY', 'RAINC', 'RAINSH', 'RAINNC', 'SNOWNC', 'GRAUPELNC', 'HAILNC', 'REFL_10CM', 'CLDFRA', 'SWDOWN', 
'GLW', 'SWNORM', 'ACSWUPT', 'ACSWUPTC', 'ACSWDNT', 'ACSWDNTC', 'ACSWUPB', 'ACSWUPBC', 'ACSWDNB', 'ACSWDNBC', 
'ACLWUPT', 'ACLWUPTC', 'ACLWDNT', 'ACLWDNTC', 'ACLWUPB', 'ACLWUPBC', 'ACLWDNB', 'ACLWDNBC', 'SWUPT', 'SWUPTC', 
'SWDNT', 'SWDNTC', 'SWUPB', 'SWUPBC', 'SWDNB', 'SWDNBC', 'LWUPT', 'LWUPTC', 'LWDNT', 'LWDNTC', 'LWUPB', 'LWUPBC', 
'LWDNB', 'LWDNBC', 'OLR', 'XLAT_U', 'XLONG_U', 'XLAT_V', 'XLONG_V', 'ALBEDO', 'CLAT', 'ALBBCK', 'EMISS', 'NOAHRES',
 'TMN', 'XLAND', 'UST', 'PBLH', 'HFX', 'QFX', 'LH', 'ACHFX', 'ACLHF', 'SNOWC', 'SR', 'SAVE_TOPO_FROM_REAL', 'REFD_MAX', 
 'ISEEDARR_SPPT', 'ISEEDARR_SKEBS', 'ISEEDARR_RAND_PERTURB', 'ISEEDARRAY_SPP_CONV', 'ISEEDARRAY_SPP_PBL', 'ISEEDARRAY_SPP_LSM', 
 'C1H', 'C2H', 'C1F', 'C2F', 'C3H', 'C4H', 'C3F', 'C4F', 'PCB', 'PC', 'LANDMASK', 'LAKEMASK', 'SST', 'SST_INPUT'])

dict_keys(['Times', 'XLAT', 'XLONG', 'LU_INDEX', 'ZNU', 'ZNW', 'ZS', 'DZS', 'VAR_SSO', 
'U', 'V', 'W', 'PH', 'PHB', 'T', 'THM', 'HFX_FORCE', 'LH_FORCE', 'TSK_FORCE', 'HFX_FORCE_TEND', 'LH_FORCE_TEND', 'TSK_FORCE_TEND', 
'MU', 'MUB', 'NEST_POS', 'P', 'PB', 'FNM', 'FNP', 'RDNW', 'RDN', 'DNW', 'DN', 'CFN', 'CFN1', 'THIS_IS_AN_IDEAL_RUN', 'P_HYD', 'Q2', 
'T2', 'TH2', 'PSFC', 'U10', 'V10', 'LPI', 'RDX', 'RDY', 'RESM', 'ZETATOP', 'CF1', 'CF2', 'CF3', 'ITIMESTEP', 'XTIME', 'QVAPOR', 'QCLOUD', 
'QRAIN', 'QICE', 'QSNOW', 'QGRAUP', 'QNICE', 'QNRAIN', 'SHDMAX', 'SHDMIN', 'SNOALB', 'TSLB', 'SMOIS', 'SH2O', 'SMCREL', 'SEAICE', 'XICEM', 
'SFROFF', 'UDROFF', 'IVGTYP', 'ISLTYP', 'VEGFRA', 'GRDFLX', 'ACGRDFLX', 'ACSNOM', 'SNOW', 'SNOWH', 'CANWAT', 'SSTSK', 'COSZEN', 'LAI', 'DTAUX3D', 
'DTAUY3D', 'DUSFCG', 'DVSFCG', 'VAR', 'CON', 'OA1', 'OA2', 'OA3', 'OA4', 'OL1', 'OL2', 'OL3', 'OL4', 'TKE_PBL', 'EL_PBL', 'MAPFAC_M', 'MAPFAC_U', 
'MAPFAC_V', 'MAPFAC_MX', 'MAPFAC_MY', 'MAPFAC_UX', 'MAPFAC_UY', 'MAPFAC_VX', 'MF_VX_INV', 'MAPFAC_VY', 'F', 'E', 'SINALPHA', 'COSALPHA', 'HGT', 'TSK',
 'P_TOP', 'T00', 'P00', 'TLP', 'TISO', 'TLP_STRAT', 'P_STRAT', 'MAX_MSTFX', 'MAX_MSTFY', 'RAINC', 'RAINSH', 'RAINNC', 'SNOWNC', 'GRAUPELNC', 'HAILNC', 
 'REFL_10CM', 'CLDFRA', 'SWDOWN', 'GLW', 'SWNORM', 'ACSWUPT', 'ACSWUPTC', 'ACSWDNT', 'ACSWDNTC', 'ACSWUPB', 'ACSWUPBC', 'ACSWDNB', 'ACSWDNBC', 'ACLWUPT', 
 'ACLWUPTC', 'ACLWDNT', 'ACLWDNTC', 'ACLWUPB', 'ACLWUPBC', 'ACLWDNB', 'ACLWDNBC', 'SWUPT', 'SWUPTC', 'SWDNT', 'SWDNTC', 'SWUPB', 'SWUPBC', 'SWDNB', 'SWDNBC', 
 'LWUPT', 'LWUPTC', 'LWDNT', 'LWDNTC', 'LWUPB', 'LWUPBC', 'LWDNB', 'LWDNBC', 'OLR', 'XLAT_U', 'XLONG_U', 'XLAT_V', 'XLONG_V', 'ALBEDO', 'CLAT', 'ALBBCK', 
 'EMISS', 'NOAHRES', 'TMN', 'XLAND', 'UST', 'PBLH', 'HFX', 'QFX', 'LH', 'ACHFX', 'ACLHF', 'SNOWC', 'SR', 'SAVE_TOPO_FROM_REAL', 'REFD_MAX', 'ISEEDARR_SPPT', 
 'ISEEDARR_SKEBS', 'ISEEDARR_RAND_PERTURB', 'ISEEDARRAY_SPP_CONV', 'ISEEDARRAY_SPP_PBL', 'ISEEDARRAY_SPP_LSM', 'C1H', 'C2H', 'C1F', 'C2F', 'C3H', 'C4H', 'C3F', 
 'C4F', 'PCB', 'PC', 'LANDMASK', 'LAKEMASK', 'SST', 'SST_INPUT'])
"""

xx = Dataset('/home/jacaraba/wrfout_d02_2017-06-24_00:00:00')
print(xx.variables.keys())