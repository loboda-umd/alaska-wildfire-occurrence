import os
import shutil
import filecmp
import logging
import datetime
from glob import glob
from pathlib import Path
# from itertools import repeat
from omegaconf import OmegaConf
from multiprocessing import cpu_count  # , Pool
from jinja2 import Environment, PackageLoader, select_autoescape

from wildfire_occurrence.model.config import Config
from wildfire_occurrence.model.common import read_config
from wildfire_occurrence.model.data_download.ncep_fnl import NCEP_FNL
from wildfire_occurrence.model.analysis.wrf_analysis import WRFAnalysis


class WRFPipeline(object):

    def __init__(
                self,
                config_filename: str,
                start_date: str,
                forecast_lenght: str,
                multi_node: bool = False
            ):

        # Configuration file intialization
        self.conf = read_config(config_filename, Config)
        logging.info(f'Loaded configuration from {config_filename}')

        # Set value for forecast start and end date
        self.start_date = start_date
        self.end_date = self.start_date + datetime.timedelta(
            days=forecast_lenght)
        logging.info(f'WRF start: {self.start_date}, end: {self.end_date}')

        # Generate working directories
        os.makedirs(self.conf.working_dir, exist_ok=True)
        logging.info(f'Created working directory {self.conf.working_dir}')

        # Setup working directories and dates
        self.simulation_dir = os.path.join(
            self.conf.working_dir,
            f'{self.start_date.strftime("%Y-%m-%d")}_' +
            f'{self.end_date.strftime("%Y-%m-%d")}'
        )
        os.makedirs(self.simulation_dir, exist_ok=True)
        logging.info(f'Created model output directory {self.simulation_dir}')

        # Setup data_dir
        self.data_dir = os.path.join(self.simulation_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)

        # Setup configuration directory
        self.conf_dir = os.path.join(self.simulation_dir, 'configs')
        os.makedirs(self.conf_dir, exist_ok=True)

        # Setup wps directory
        self.local_wps_path = os.path.join(self.simulation_dir, 'WPS')
        self.local_wrf_path = os.path.join(self.simulation_dir, 'em_real')
        self.local_wrf_output = os.path.join(self.simulation_dir, 'output')
        self.local_wrf_output_vars = os.path.join(
            self.simulation_dir, 'variables')

        # Setup configuration filenames
        self.wps_conf_filename = os.path.join(self.conf_dir, 'namelist.wps')
        self.wrf_conf_filename = os.path.join(self.conf_dir, 'namelist.input')

        # Setup configuration filenames local to directories
        self.local_wps_conf = os.path.join(self.local_wps_path, 'namelist.wps')
        self.local_wrf_conf = os.path.join(
            self.local_wrf_path, 'namelist.input')

        # setup multi_node variable
        self.conf.multi_node = multi_node

    # -------------------------------------------------------------------------
    # setup
    # -------------------------------------------------------------------------
    def setup(self) -> None:

        # Working on the setup of the project
        logging.info('Starting setup pipeline step')

        # Working on the setup of the project
        logging.info('Starting download from setup pipeline step')

        # Generate subdirectories to work with WRF
        os.makedirs(self.data_dir, exist_ok=True)
        logging.info(f'Created data directory {self.data_dir}')

        # Generate data downloader
        data_downloader = NCEP_FNL(
            self.data_dir,
            self.start_date,
            self.end_date
        )
        data_downloader.download()

        # Generate configuration files for WPS - namelist.wps
        self.setup_wps_config()

        # Generate configuration files for WRF - namelist.input
        self.setup_wrf_config()

        return

    # -------------------------------------------------------------------------
    # geogrid
    # -------------------------------------------------------------------------
    def geogrid(self) -> None:

        logging.info('Preparing to run geogrid.exe')

        # setup WPS directory
        if not os.path.exists(self.local_wps_path):
            shutil.copytree(
                self.conf.wps_path, self.local_wps_path, dirs_exist_ok=True)
            logging.info(f'Done copying WPS to {self.local_wps_path}')

        # create configuration file symlink
        self._symlink_conf_file(self.wps_conf_filename, self.local_wps_conf)

        # go to WPS directory and run wps
        os.chdir(self.local_wps_path)
        logging.info(f'Changed working directory to {self.local_wps_path}')

        # setup geogrid command
        if not self.conf.multi_node:
            geodrid_cmd = \
                'singularity exec -B /explore/nobackup/projects/ilab,' + \
                '$NOBACKUP,/panfs/ccds02/nobackup/projects/ilab ' + \
                f'{self.conf.container_path} ' + \
                f'mpirun -np {cpu_count()} --oversubscribe ./geogrid.exe'
        else:
            geodrid_cmd = 'mpirun -np 40 --host gpu016 --oversubscribe' + \
                'singularity exec -B /explore/nobackup/projects/ilab,' + \
                '$NOBACKUP,/lscratch,/panfs/ccds02/nobackup/projects/ilab ' + \
                f'{self.conf.container_path} ./geogrid.exe'

        # run geogrid command
        os.system(geodrid_cmd)

        return

    # -------------------------------------------------------------------------
    # ungrib
    # -------------------------------------------------------------------------
    def ungrib(self) -> None:

        logging.info('Preparing to run geogrid.exe')

        # assert WPS directory
        assert os.path.exists(self.local_wps_path), \
            f'{self.local_wps_path} does not exist, please run geogrid first.'

        # create Vtable symlink
        local_wps_vtable = os.path.join(self.local_wps_path, 'Vtable')
        if not os.path.lexists(local_wps_vtable):
            os.symlink(
                os.path.join(
                    self.local_wps_path, 'ungrib/Variable_Tables/Vtable.GFS'),
                local_wps_vtable
            )
        logging.info(f'Created Vtable symlink on {self.local_wps_path}')

        # go to WPS directory and run wps
        os.chdir(self.local_wps_path)
        logging.info(f'Changed working directory to {self.local_wps_path}')

        # find all files in directory, and extract common prefix
        common_prefix = os.path.commonprefix(
            glob(f'{self.data_dir}/{str(self.start_date.year)}/*'))

        # run link_grib
        os.system(f'./link_grib.csh {common_prefix}')
        logging.info('Done with link_grib.csh')

        # setup ungrib command
        if not self.conf.multi_node:
            ungrib_cmd = \
                'singularity exec -B /explore/nobackup/projects/ilab,' + \
                '$NOBACKUP,/panfs/ccds02/nobackup/projects/ilab ' + \
                f'{self.conf.container_path} ./ungrib.exe'
        else:
            ungrib_cmd = \
                'srun --mpi=pmix -N 1 -n 1 singularity exec -B ' + \
                '/explore/nobackup/projects/ilab,' + \
                '$NOBACKUP,/panfs/ccds02/nobackup/projects/ilab ' + \
                f'{self.conf.container_path} ' + \
                './ungrib.exe'

        # run ungrib command
        os.system(ungrib_cmd)

        return

    # -------------------------------------------------------------------------
    # metgrid
    # -------------------------------------------------------------------------
    def metgrid(self) -> None:

        logging.info('Preparing to run metgrid.exe')

        # assert WPS directory
        assert os.path.exists(self.local_wps_path), \
            f'{self.local_wps_path} does not exist, please run geogrid first.'

        # go to WPS directory and run wps
        os.chdir(self.local_wps_path)
        logging.info(f'Changed working directory to {self.local_wps_path}')

        # setup metgrid command
        if not self.conf.multi_node:
            metgrid_cmd = \
                'singularity exec -B /explore/nobackup/projects/ilab,' + \
                '$NOBACKUP,/panfs/ccds02/nobackup/projects/ilab ' + \
                f'{self.conf.container_path} ' + \
                f'mpirun -np {cpu_count()} --oversubscribe ./metgrid.exe'
        else:
            metgrid_cmd = \
                f'srun --mpi=pmix -N 1 -n {cpu_count()} singularity ' + \
                'exec -B /explore/nobackup/projects/ilab,' + \
                '$NOBACKUP,/panfs/ccds02/nobackup/projects/ilab ' + \
                f'{self.conf.container_path} ' + \
                './metgrid.exe'

        # run metgrid command
        os.system(metgrid_cmd)

        return

    # -------------------------------------------------------------------------
    # real
    # -------------------------------------------------------------------------
    def real(self) -> None:

        logging.info('Preparing to run em_real.exe')

        # setup WRF em_real directory
        if not os.path.exists(self.local_wrf_path):
            shutil.copytree(
                os.path.join(self.conf.wrf_path, 'test', 'em_real'),
                self.local_wrf_path, dirs_exist_ok=True)
            logging.info(f'Done copying WRF em_real to {self.local_wrf_path}')

        # move the created met_em* files to the em_real directory
        for met_filename in glob(os.path.join(self.local_wps_path, 'met_em*')):
            local_met_filename = os.path.join(
                    self.local_wrf_path, f'{Path(met_filename).stem}.nc')
            if not os.path.exists(local_met_filename):
                os.symlink(met_filename, local_met_filename)

        # create configuration file symlink
        self._symlink_conf_file(self.wrf_conf_filename, self.local_wrf_conf)

        # go to WRF directory and run real
        os.chdir(self.local_wrf_path)
        logging.info(f'Changed working directory to {self.local_wrf_path}')

        # setup real command
        if not self.conf.multi_node:
            real_cmd = \
                'singularity exec -B /explore/nobackup/projects/ilab,' + \
                '$NOBACKUP,/panfs/ccds02/nobackup/projects/ilab ' + \
                f'{self.conf.container_path} ' + \
                f'mpirun -np {cpu_count()} --oversubscribe ./real.exe'
        else:
            real_cmd = \
                'srun --mpi=pmix -N 2 -n 80 singularity ' + \
                'exec -B /explore/nobackup/projects/ilab,' + \
                '$NOBACKUP,/panfs/ccds02/nobackup/projects/ilab ' + \
                f'{self.conf.container_path} ' + \
                './real.exe'

        # run metgrid command
        os.system(real_cmd)

        return

    # -------------------------------------------------------------------------
    # wrf
    # -------------------------------------------------------------------------
    def wrf(self) -> None:

        logging.info('Preparing to run wrf.exe')

        # assert WPS directory
        assert os.path.exists(self.local_wrf_path), \
            f'{self.local_wrf_path} does not exist, please run real first.'

        # go to WPS directory and run wps
        os.chdir(self.local_wrf_path)
        logging.info(f'Changed working directory to {self.local_wrf_path}')

        # setup metgrid command
        if not self.conf.multi_node:
            wrf_cmd = \
                'singularity exec -B /explore/nobackup/projects/ilab,' + \
                '$NOBACKUP,/panfs/ccds02/nobackup/projects/ilab ' + \
                f'{self.conf.container_path} ' + \
                f'mpirun -np {cpu_count()} --oversubscribe ./wrf.exe'
        else:
            wrf_cmd = \
                'srun --mpi=pmix -N 2 -n 80 singularity ' + \
                'exec -B /explore/nobackup/projects/ilab,' + \
                '$NOBACKUP,/panfs/ccds02/nobackup/projects/ilab ' + \
                f'{self.conf.container_path} ' + \
                './wrf.exe'

        # run metgrid command
        os.system(wrf_cmd)

        # TODO
        # move output files at the end to something like working_dir/results

        return

    # -------------------------------------------------------------------------
    # postprocess
    # -------------------------------------------------------------------------
    def postprocess(self) -> None:

        logging.info('Preparing to postprocess and extract variables')

        # create output directory
        os.makedirs(self.local_wrf_output, exist_ok=True)
        logging.info(f'Created WRF output directory {self.local_wrf_output}')

        # if the files have not been moved, move them to the output dir
        if len(os.listdir(self.local_wrf_output)) == 0:

            # get filenames, make sure they exist
            wrf_output_filenames = \
                glob(os.path.join(self.local_wrf_path, 'auxhist24_d0*')) + \
                glob(os.path.join(self.local_wrf_path, 'wrfout_d0*'))
            assert len(wrf_output_filenames) > 0, \
                'WRF output (auxhist24_d0*, wrfout_d0*) not found. Re-run WRF.'

            # move output to clean directory
            for filename in wrf_output_filenames:
                shutil.move(filename, self.local_wrf_output)
            logging.info(f'Moved WRF output to {self.local_wrf_output}')

        # Get WRF output filename
        wrf_output_filename = glob(
            os.path.join(self.local_wrf_output, self.conf.wrf_output_filename))
        assert len(wrf_output_filename) == 1, \
            f'WRF output filename not found under {self.local_wrf_output}.'

        # Get first item from the list
        wrf_output_filename = wrf_output_filename[0]
        logging.info(f'Loading {wrf_output_filename}')

        # create WRFAnalysis object, stores wrf_dataset
        wrf_analysis = WRFAnalysis(wrf_output_filename)

        # variables output dir
        os.makedirs(self.local_wrf_output_vars, exist_ok=True)
        logging.info(
            f'Created WRF vars output directory {self.local_wrf_output_vars}')

        """
        # parallel extraction of variables
        p = Pool(processes=cpu_count())
        p.starmap(
            self._compute_variables,
            zip(
                range(len(wrf_analysis.wrf_dataset.Times.values)),
                wrf_analysis.wrf_dataset.Times.values,
                repeat(wrf_output_filename)
            )
        )
        """

        # serial extraction of variables
        for t_idx, delta_time in \
                enumerate(wrf_analysis.wrf_dataset.Times.values):

            logging.info(f'Processing t_idx: {t_idx}, timestamp: {delta_time}')

            # setup output filename
            output_filename = os.path.join(
                    self.local_wrf_output_vars,
                    f"d02_{delta_time.astype(str).replace(':', '-')}.tif")

            # if the imagery does not exist
            if not os.path.isfile(output_filename):

                # compute WRF variables and output to disk
                wrf_analysis.compute_all_and_write(
                    timeidx=t_idx,
                    output_variables=OmegaConf.to_object(
                        self.conf.wrf_output_variables),
                    output_filename=output_filename
                )

    # -------------------------------------------------------------------------
    # _compute_variables
    # -------------------------------------------------------------------------
    def _compute_variables(self, t_idx, delta_time, wrf_output_filename):

        logging.info(f'Processing t_idx: {t_idx}, timestamp: {delta_time}')

        # setup output filename
        output_filename = os.path.join(
                self.local_wrf_output_vars,
                f"d02_{delta_time.astype(str).replace(':', '-')}.tif")

        # if the imagery does not exist
        if not os.path.isfile(output_filename):

            # unfortunately, netCDF object is not pickable, thus we need
            # to redifine it on every single process, hopefully there
            # will be an implementation on it at some point by them
            wrf_analysis = WRFAnalysis(wrf_output_filename)

            # compute WRF variables and output to disk
            wrf_analysis.compute_all_and_write(
                timeidx=t_idx,
                output_variables=OmegaConf.to_object(
                        self.conf.wrf_output_variables),
                output_filename=output_filename
            )
        return

    # -------------------------------------------------------------------------
    # setup_wps_config
    # -------------------------------------------------------------------------
    def setup_wps_config(self, template_filename: str = 'namelist.wps.jinja2'):

        # Setup jinja2 Environment
        env = Environment(
            loader=PackageLoader("wildfire_occurrence"),
            autoescape=select_autoescape()
        )

        # Get the template of the environment for WPS
        template = env.get_template(template_filename)

        # Modify configuration to include start and end date
        self.conf.wps_config['start_date'] = \
            self.start_date.strftime("%Y-%m-%d_%H:%M:%S")
        self.conf.wps_config['end_date'] = \
            self.end_date.strftime("%Y-%m-%d_%H:%M:%S")

        # Fill in elements from the WPS environment and save filename
        template.stream(self.conf.wps_config).dump(self.wps_conf_filename)
        logging.info(f'Saved WPS configuration at {self.wps_conf_filename}')

        return

    # -------------------------------------------------------------------------
    # setup_wrf_config
    # -------------------------------------------------------------------------
    def setup_wrf_config(
                self, template_filename: str = 'namelist.input.jinja2'):

        # Setup jinja2 Environment
        env = Environment(
            loader=PackageLoader("wildfire_occurrence"),
            autoescape=select_autoescape()
        )

        # Get the template of the environment for WPS
        template = env.get_template(template_filename)

        # Modify configuration to include start and end date
        self.conf.wrf_config['start_year'] = self.start_date.year
        self.conf.wrf_config['end_year'] = self.end_date.year
        self.conf.wrf_config['start_month'] = self.start_date.strftime('%m')
        self.conf.wrf_config['end_month'] = self.end_date.strftime('%m')
        self.conf.wrf_config['start_day'] = self.start_date.strftime('%d')
        self.conf.wrf_config['end_day'] = self.end_date.strftime('%d')
        self.conf.wrf_config['start_hour'] = self.start_date.strftime('%H')
        self.conf.wrf_config['end_hour'] = self.end_date.strftime('%H')

        # Fill in elements from the WRF environment and save filename
        template.stream(self.conf.wrf_config).dump(self.wrf_conf_filename)
        logging.info(f'Saved WRF configuration at {self.wrf_conf_filename}')

        return

    def _symlink_conf_file(self, source, destination):

        # condition 1: local configuration file does not exist, symlink it
        if not os.path.lexists(destination):

            # make sure configuration file exists
            assert os.path.isfile(source), \
                'Please run setup pipeline step before running real'

            # add symlink
            os.symlink(source, destination)

            logging.info(f'Created namelist.input symlink at {destination}')

        # condition #2: local configuration exists, but is not the latest one
        elif not filecmp.cmp(source, destination):

            # remove previous version
            os.remove(destination)

            # add symlink
            os.symlink(source, destination)

            logging.info(
                f'Removed old copy of {destination}. Created new '
                f'namelist.input symlink at {destination}')

        else:

            logging.info(
                f'namelist.input exists, {destination}, '
                'nothing to copy.')
