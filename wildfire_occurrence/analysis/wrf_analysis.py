import os
import xwrf
import netCDF4
import logging
import xarray as xr
from glob import glob
from rasterio.crs import CRS
from wrf import getvar, interplevel

__all__ = ["xwrf"]


class WRFAnalysis(object):

    def __init__(self, input_filename_regex):

        # wrf input_filename_regex
        self.wrf_filename_regex = input_filename_regex

        # get list of wrf input filenames
        self.wrf_filenames = sorted(glob(self.wrf_filename_regex))

        # get dataset into xr using xwrf format
        self.wrf_dataset = xr.open_mfdataset(
            self.wrf_filenames,
            engine="netcdf4",
            parallel=True,
            concat_dim="Time",
            combine="nested",
            chunks=None,
            decode_times=False,
            decode_coords="all",
        ).xwrf.postprocess(drop_diagnostic_variable_components=False)

        # get CRS from xwrf consolidation
        self.crs = CRS.from_string(
            str(self.wrf_dataset['wrf_projection'].values))

        # assign crs to override crs=None and to be compliant with rioxarray
        self.wrf_dataset.rio.write_crs(self.crs, inplace=True)

        # get netCDF objects compatible with wrf-python
        # this is needed since wrf-python does not accept xwrf input
        self.wrf_python_dataset = [
            netCDF4.Dataset(f) for f in self.wrf_filenames]

        # get list of variables, and remove grid_mapping attribute
        # this is needed to be compliant with rioxarray
        vars_list = list(self.wrf_dataset.data_vars)
        for var in vars_list:
            if 'grid_mapping' in self.wrf_dataset[var].attrs:
                del self.wrf_dataset[var].attrs['grid_mapping']

    def compute_all_and_write(
                self,
                timeidx: int = 0,
                output_variables: list = ["LANDMASK"],
                output_filename: str = None,
                nodata: int = -10001
            ):
        """
        We use this function to compute lightning specific variables
        and to store them in the same dataset.
        """

        """
        ['Times', 'LU_INDEX', 'ZS', 'DZS', 'VAR_SSO', 'BATHYMETRY_FLAG',
        'U', 'V', 'W','PH', 'PHB', 'T', 'THM', 'HFX_FORCE', 'LH_FORCE',
        'TSK_FORCE', 'HFX_FORCE_TEND','LH_FORCE_TEND', 'TSK_FORCE_TEND',
        'MU', 'MUB', 'NEST_POS', 'P', 'PB', 'FNM', 'FNP','RDNW', 'RDN',
        'DNW', 'DN', 'CFN', 'CFN1', 'THIS_IS_AN_IDEAL_RUN', 'P_HYD', 'Q2',
        'T2', 'TH2', 'PSFC', 'U10', 'V10', 'LPI', 'RDX', 'RDY', 'AREA2D',
        'DX2D', 'RESM','ZETATOP', 'CF1', 'CF2', 'CF3', 'ITIMESTEP', 'QVAPOR',
        'QCLOUD', 'QRAIN', 'QICE','QSNOW', 'QGRAUP', 'QNICE', 'QNRAIN',
        'SHDMAX','SHDMIN', 'SNOALB', 'TSLB', 'SMOIS','SH2O', 'SMCREL',
        'SEAICE', 'XICEM', 'SFROFF', 'UDROFF', 'IVGTYP', 'ISLTYP', 'VEGFRA',
        'GRDFLX', 'ACGRDFLX', 'ACSNOM', 'SNOW', 'SNOWH', 'CANWAT', 'SSTSK',
        'WATER_DEPTH', 'COSZEN', 'LAI', 'U10E', 'V10E', 'DTAUX3D', 'DTAUY3D',
        'DUSFCG', 'DVSFCG', 'VAR', 'CON', 'OA1', 'OA2', 'OA3', 'OA4', 'OL1',
        'OL2', 'OL3', 'OL4', 'TKE_PBL', 'EL_PBL', 'O3_GFS_DU', 'MAPFAC_M',
        'MAPFAC_U', 'MAPFAC_V', 'MAPFAC_MX', 'MAPFAC_MY', 'MAPFAC_UX',
        'MAPFAC_UY', 'MAPFAC_VX', 'MF_VX_INV', 'MAPFAC_VY', 'F', 'E',
        'SINALPHA','COSALPHA', 'HGT', 'TSK', 'P_TOP', 'GOT_VAR_SSO', 'T00',
        'P00', 'TLP','TISO', 'TLP_STRAT', 'P_STRAT', 'MAX_MSFTX', 'MAX_MSFTY',
        'RAINC','RAINSH', 'RAINNC', 'SNOWNC', 'GRAUPELNC', 'HAILNC',
        'REFL_10CM','CLDFRA','SWDOWN', 'GLW', 'SWNORM', 'ACSWUPT', 'ACSWUPTC',
        'ACSWDNT', 'ACSWDNTC','ACSWUPB', 'ACSWUPBC', 'ACSWDNB', 'ACSWDNBC',
        'ACLWUPT', 'ACLWUPTC', 'ACLWDNT','ACLWDNTC', 'ACLWUPB', 'ACLWUPBC',
        'ACLWDNB', 'ACLWDNBC', 'SWUPT', 'SWUPTC', 'SWDNT', 'SWDNTC', 'SWUPB',
        'SWUPBC', 'SWDNB', 'SWDNBC', 'LWUPT', 'LWUPTC', 'LWDNT', 'LWDNTC',
        'LWUPB', 'LWUPBC', 'LWDNB', 'LWDNBC', 'OLR', 'ALBEDO', 'ALBBCK',
        'EMISS', 'NOAHRES', 'TMN', 'XLAND', 'UST', 'PBLH', 'HFX', 'QFX',
        'LH', 'ACHFX', 'ACLHF', 'SNOWC', 'SR', 'SAVE_TOPO_FROM_REAL',
        'REFD_MAX', 'ISEEDARR_SPPT', 'ISEEDARR_SKEBS', 'ISEEDARR_RAND_PERTURB',
        'ISEEDARRAY_SPP_CONV', 'ISEEDARRAY_SPP_PBL', 'ISEEDARRAY_SPP_LSM',
        'C1H', 'C2H', 'C1F', 'C2F', 'C3H', 'C4H', 'C3F', 'C4F', 'PCB', 'PC',
        'LANDMASK', 'LAKEMASK', 'SST', 'SST_INPUT', 'air_potential_temperature'
        'air_pressure', 'geopotential', 'geopotential_height', 'wind_east',
        'wind_north']
        """

        # create a copy of the dataset with a single time step
        wrf_dataset_single_time = self.wrf_dataset.isel(Time=timeidx)

        # compute LPI - LPI is already computed by xwrf

        # compute Helicity
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            Helicity=self.compute_var('helicity', timeidx))

        # compute LCL, given by CAPE
        # wrf_dataset_single_time = wrf_dataset_single_time.assign(
        #    LCL=self.compute_var('lcl', timeidx))

        # compute PW
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            PW=self.compute_var('pw', timeidx))

        # compute SLP
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            SLP=self.compute_var('slp', timeidx))

        # compute GPZ levels
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            GPZ500=self.compute_gpz(500, timeidx))
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            GPZ700=self.compute_gpz(700, timeidx))
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            GPZ750=self.compute_gpz(750, timeidx))
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            GPZ850=self.compute_gpz(850, timeidx))
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            GPZ1000=self.compute_gpz(1000, timeidx))

        # compute DZ levels
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            DZ500_1000=self.compute_dz(
                wrf_dataset_single_time['GPZ500'],
                wrf_dataset_single_time['GPZ1000']
            )
        )
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            DZ850_1000=self.compute_dz(
                wrf_dataset_single_time['GPZ850'],
                wrf_dataset_single_time['GPZ1000']
            )
        )
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            DZ700_850=self.compute_dz(
                wrf_dataset_single_time['GPZ700'],
                wrf_dataset_single_time['GPZ850']
            )
        )
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            DZ700_1000=self.compute_dz(
                wrf_dataset_single_time['GPZ700'],
                wrf_dataset_single_time['GPZ1000']
            )
        )

        # compute RH2
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            RH2=self.compute_var('rh2', timeidx))

        # compute RH levels
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            RH500=self.compute_rh(500, timeidx))
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            RH700=self.compute_rh(700, timeidx))
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            RH800=self.compute_rh(800, timeidx))
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            RH850=self.compute_rh(850, timeidx))

        # compute T2 - T2 is already computed by xwrf

        # compute Td2
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            TD2=self.compute_var('td2', timeidx))

        # compute TD levels
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            TD500=self.compute_td(500, timeidx))
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            TD700=self.compute_td(700, timeidx))
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            TD850=self.compute_td(850, timeidx))
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            TD1000=self.compute_td(1000, timeidx))

        # compute TC levels
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            TC500=self.compute_tc(500, timeidx))
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            TC700=self.compute_tc(700, timeidx))
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            TC850=self.compute_tc(850, timeidx))

        # compute TP levels, double-check equation for this one
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            TP500=self.compute_tp(500, timeidx))
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            TP850=self.compute_tp(850, timeidx))

        # compute SHOW
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            SHOW=self.compute_show(
                wrf_dataset_single_time['TC500'],
                wrf_dataset_single_time['TP850']
            )
        )

        # compute TT
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            TT=self.compute_tt(
                wrf_dataset_single_time['TC850'],
                wrf_dataset_single_time['TD850'],
                wrf_dataset_single_time['TC500']
            )
        )

        # compute Rain
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            RAINTotal=self.compute_rain(
                wrf_dataset_single_time['RAINNC'],
                wrf_dataset_single_time['RAINC']
            )
        )

        # compute W levels
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            W500=self.compute_w(500, timeidx))

        # compute WA levels
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            WA500=self.compute_wa(500, timeidx))

        # compute cloud frac levels
        cloud_frac_variables = self.compute_cloudfrac(timeidx)
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            CFLow=cloud_frac_variables[0])
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            CFMed=cloud_frac_variables[1])
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            CFHigh=cloud_frac_variables[2])
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            CFTotal=cloud_frac_variables[3])

        # compute CAPE2D variables: MCAPE, MCIN, LCL, and LFC.
        cape_variables = self.compute_cape2d(timeidx)
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            MCAPE=cape_variables[0])
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            MCIN=cape_variables[1])
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            LCL=cape_variables[2])
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            LFC=cape_variables[3])

        # compute T levels
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            T500=self.compute_t(500, timeidx))
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            T750=self.compute_t(750, timeidx))
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            T850=self.compute_t(850, timeidx))

        # PLI IS INCORRECT, WE NEED TO FIX THIS
        # compute PLI, might be incorrectly computed, need to double check
        # https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.lifted_index.html
        wrf_dataset_single_time = wrf_dataset_single_time.assign(
            PLI=self.compute_pli(
                wrf_dataset_single_time['T500'],
                wrf_dataset_single_time['TP500']
            )
        )

        # Proceed to write the variables into the dataset
        wrf_dataset_single_time[output_variables].rio.write_crs(
            self.crs, inplace=True)

        wrf_dataset_single_time = wrf_dataset_single_time[
            output_variables].to_array()
        wrf_dataset_single_time.rio.write_nodata(nodata, inplace=True)
        wrf_dataset_single_time.attrs['long_name'] = output_variables

        if output_filename is not None:
            wrf_dataset_single_time.rio.to_raster(
                output_filename,
                BIGTIFF='IF_SAFER',
                compress='LZW',
                driver='GTiff',
                dtype='float32',
                recalc_transform=False
            )

        return wrf_dataset_single_time

    def compute_var(self, var_name: str, timeidx: int = 0):
        var_output = getvar(
            self.wrf_python_dataset, var_name, timeidx=timeidx)
        var_output = var_output.rio.write_crs(self.crs, inplace=True)
        return var_output.rename({'south_north': 'y', 'west_east': 'x'})

    def compute_gpz(self, pressure_level: int = 500, timeidx: int = 0):
        pressure = getvar(
            self.wrf_python_dataset, 'pressure', timeidx=timeidx)
        gpz = getvar(
            self.wrf_python_dataset, 'geopt', timeidx=timeidx) / 9.81
        var_output = interplevel(gpz, pressure, pressure_level)
        var_output = var_output.rio.write_crs(self.crs, inplace=True)
        return var_output.rename({'south_north': 'y', 'west_east': 'x'})

    def compute_rh(self, pressure_level: int = 500, timeidx: int = 0):
        pressure = getvar(
            self.wrf_python_dataset, 'pressure', timeidx=timeidx)
        rh = getvar(
            self.wrf_python_dataset, 'rh', timeidx=timeidx)
        var_output = interplevel(rh, pressure, pressure_level)
        var_output = var_output.rio.write_crs(self.crs, inplace=True)
        return var_output.rename({'south_north': 'y', 'west_east': 'x'})

    def compute_td(self, pressure_level: int = 500, timeidx: int = 0):
        pressure = getvar(
            self.wrf_python_dataset, 'pressure', timeidx=timeidx)
        td = getvar(
            self.wrf_python_dataset, 'td', timeidx=timeidx)
        var_output = interplevel(td, pressure, pressure_level)
        var_output = var_output.rio.write_crs(self.crs, inplace=True)
        return var_output.rename({'south_north': 'y', 'west_east': 'x'})

    def compute_t(self, pressure_level: int = 500, timeidx: int = 0):
        pressure = getvar(
            self.wrf_python_dataset, 'pressure', timeidx=timeidx)
        t = getvar(
            self.wrf_python_dataset, 'T', timeidx=timeidx)
        var_output = interplevel(t, pressure, pressure_level)
        var_output = var_output.rio.write_crs(self.crs, inplace=True)
        return var_output.rename({'south_north': 'y', 'west_east': 'x'})

    def compute_tc(self, pressure_level: int = 500, timeidx: int = 0):
        pressure = getvar(
            self.wrf_python_dataset, 'pressure', timeidx=timeidx)
        tc = getvar(
            self.wrf_python_dataset, 'tc', timeidx=timeidx)
        var_output = interplevel(tc, pressure, pressure_level)
        var_output = var_output.rio.write_crs(self.crs, inplace=True)
        return var_output.rename({'south_north': 'y', 'west_east': 'x'})

    def compute_tp(self, pressure_level: int = 500, timeidx: int = 0):
        # consider removing this extract calculation, try to find it
        # if its already computed
        pressure = getvar(
            self.wrf_python_dataset, 'pressure', timeidx=timeidx)
        tc = getvar(
            self.wrf_python_dataset, 'tc', timeidx=timeidx)
        tc_interpolated = interplevel(tc, pressure, pressure_level)
        var_output = \
            (tc_interpolated + 273.15) * \
            ((500. / pressure_level)**0.286) - 273.15
        # (tc_850 + 273.15)*((500/850)^0.286) - 273.15
        var_output = var_output.rio.write_crs(self.crs, inplace=True)
        return var_output.rename({'south_north': 'y', 'west_east': 'x'})

    def compute_dz(self, gpz1, gpz2):
        return gpz1 - gpz2

    def compute_show(self, tc_500, tp_850):
        return tc_500 - tp_850

    def compute_tt(self, tc_850, td_850, tc_500):
        return tc_850 + td_850 - 2 * tc_500

    def compute_rain(self, rain_exp, rain_con):
        return rain_exp + rain_con

    def compute_w(self, pressure_level: int = 500, timeidx: int = 0):
        pressure = getvar(
            self.wrf_python_dataset, 'pressure', timeidx=timeidx)
        w = getvar(
            self.wrf_python_dataset, 'W', timeidx=timeidx)
        w = w[:pressure.shape[0], :, :]
        var_output = interplevel(w, pressure, pressure_level)
        var_output = var_output.rio.write_crs(self.crs, inplace=True)
        return var_output.rename({'south_north': 'y', 'west_east': 'x'})

    def compute_wa(self, pressure_level: int = 500, timeidx: int = 0):
        pressure = getvar(
            self.wrf_python_dataset, 'pressure', timeidx=timeidx)
        wa = getvar(
            self.wrf_python_dataset, 'wa', timeidx=timeidx)
        var_output = interplevel(wa, pressure, pressure_level)
        var_output = var_output.rio.write_crs(self.crs, inplace=True)
        return var_output.rename({'south_north': 'y', 'west_east': 'x'})

    def compute_cloudfrac(self, timeidx: int = 0):
        cloudfrac = getvar(
            self.wrf_python_dataset, "cloudfrac", timeidx=timeidx)
        cloudfrac = cloudfrac.rename({'south_north': 'y', 'west_east': 'x'})
        cloudfrac = cloudfrac.rio.write_crs(self.crs, inplace=True)
        low_cloudfrac = cloudfrac[0, :, :]
        mid_cloudfrac = cloudfrac[1, :, :]
        high_cloudfrac = cloudfrac[2, :, :]
        total_cloudfrac = (low_cloudfrac + mid_cloudfrac + high_cloudfrac) / 3
        return low_cloudfrac, mid_cloudfrac, high_cloudfrac, total_cloudfrac

    def compute_cape2d(self, timeidx: int = 0):
        cape2d = getvar(
            self.wrf_python_dataset, "cape_2d", timeidx=timeidx)
        cape2d = cape2d.rename({'south_north': 'y', 'west_east': 'x'})
        cape2d = cape2d.rio.write_crs(self.crs, inplace=True)
        mcape = cape2d[0, :, :]
        mcin = cape2d[1, :, :]
        lcl = cape2d[2, :, :]
        lfc = cape2d[3, :, :]

        return mcape, mcin, lcl, lfc

    def compute_pli(self, t_500, tp_500):
        return t_500 - tp_500


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    filename_regex = \
        '/explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/' + \
        'WRF_Simulations/2022-07-03/wrfout_d02*'

    filename_regex = \
        '/explore/nobackup/projects/ilab/projects/LobodaTFO/operations/' + \
        '2023-06-24_2023-07-04/output/wrfout_d02*'
    data_filenames = glob(filename_regex)

    for filename in data_filenames:

        # /explore/nobackup/projects/ilab/projects/LobodaTFO/data/WRF_Data/WRF_Simulations/*/wrfout_d02*
        # create WRFAnalysis object, stores wrf_dataset
        wrf_analysis = WRFAnalysis(filename)

        # output variables for the lightning model
        # these are the variables of importance between both 24 and 48 models
        # output_variables = [
        #    'PLI', 'GPZ500', 'GPZ700', 'TD500', 'CFTotal',
        #    'RH500', 'SLP', 'W500', 'RH700', 'CFLow', 'TD2',
        #    'TT', 'Helicity', 'GPZ850', 'SHOW', 'LCL',
        #    'RH2', 'T850', 'RH850', 'Rain', 'T2', 'DZ700_850',
        #    'RH800', 'T500', 'PW', 'T750'
        # ] # BT missing

        output_variables = [
            'CFTotal', 'CFLow', 'CFMed', 'CFHigh',
            'DZ700_850',
            'GPZ500', 'GPZ700', 'GPZ750', 'GPZ850',
            'Helicity',
            'LCL', 'LFC',
            'MCAPE', 'MCIN',
            'PLI', 'PW',
            'RAINTotal',
            'RH2', 'RH500', 'RH700', 'RH800', 'RH850',
            'SHOW',
            'SLP',
            'TD2', 'TD500',
            'TT', 'T2', 'T500', 'T750', 'T850',
            'W500', 'WA500'
        ]  # BT missing

        # looks good - Helicity, SLP, 'GPZ500', 'TD500', 'RH500',
        # 'TD2', 'LCL', 'PW', 'RH2', 'RAINTotal'
        # TT, TC500
        # wrong - PLI
        # maybe
        #   CFTotoal (wrong, local is 0, ours is higher)
        #   CFlow (wrong, local is 0, ours is higher)
        #   CFMed  (wrong, local is 0, ours is higher)
        #   CFHigh (wrong, local is 0, ours is higher)
        #   'GPZ750' no-data problems, kind of similar
        #   'GPZ700' no local data to compare to
        #   'RH700' no local data to compare to
        #   'GPZ850' no local data to compare to
        #   'SHOW' wrong because of no data
        #   'T500' no local data to compare to
        #    'RH800' wrong because of no data
        #    'T2' numbers look far away - our T2 is in Kelvin
        #    'RH850' no local data to compare to
        #    'T850' no local data to compare to
        #   'W500'
        #   'WA500' -looks good
        #  'DZ700_850' - looks good
        # 'BT' missing

        output_dir = 'output'  # os.path.dirname(os.path.abspath(filename))

        # TODO: make this for loop parallel later
        for t_idx, delta_time in enumerate(
                wrf_analysis.wrf_dataset.Times.values):

            logging.info(f'Processing t_idx: {t_idx}, timestamp: {delta_time}')
            output_filename = os.path.join(
                output_dir,
                f"d02_{delta_time.astype(str).replace(':', '-')}.tif")

            if not os.path.isfile(output_filename):

                wrf_analysis.compute_all_and_write(
                    timeidx=t_idx,
                    output_variables=output_variables,
                    output_filename=output_filename
                )
