import os
import shutil
import logging
import datetime
from jinja2 import Environment, PackageLoader, select_autoescape

from wildfire_occurrence.model.config import Config
from wildfire_occurrence.model.common import read_config
from wildfire_occurrence.model.data_download.ncep_fnl import NCEP_FNL


class WRFPipeline(object):

    def __init__(
                self,
                config_filename: str,
                start_date: str,
                forecast_lenght: str
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
            f'{self.start_date.strftime("%Y-%m-%d")}'
        )
        os.makedirs(self.simulation_dir, exist_ok=True)
        logging.info(f'Created output directory {self.simulation_dir}')

        # Setup data_dir
        self.data_dir = os.path.join(self.simulation_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)

        # Setup configuration directory
        self.conf_dir = os.path.join(self.simulation_dir, 'configs')
        os.makedirs(self.conf_dir, exist_ok=True)

        # Setup configuration filenames
        self.wps_conf_filename = os.path.join(self.conf_dir, 'namelist.wps')
        self.wrf_conf_filename = os.path.join(self.conf_dir, 'namelist.input')

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

        # Generate configuration files for WRF - namelist.wps
        self.setup_wps_config()

        # Generate configuration files for WRF - namelist.input

        return

    # -------------------------------------------------------------------------
    # geogrid
    # -------------------------------------------------------------------------
    def geogrid(self) -> None:

        logging.info('Preparing to run geogrid.exe')

        # setup WPS directory
        local_wps_path = os.path.join(self.simulation_dir, 'WPS')
        if not os.path.exists(local_wps_path):
            shutil.copytree(
                self.conf.wps_path, local_wps_path, dirs_exist_ok=True)
            logging.info(f'Done copying WPS to {local_wps_path}')

        # create configuration file symlink
        local_wps_conf = os.path.join(local_wps_path, 'namelist.wps')
        if not os.path.lexists(local_wps_conf):
            os.symlink(
                self.wps_conf_filename,
                local_wps_conf
            )
        logging.info(f'Created namelist.wps symlink on {local_wps_path}')

        # go to WPS directory and run wps
        os.chdir(local_wps_path)
        logging.info(f'Changed working directory to {local_wps_path}')

        # setup geogrid command
        geodrid_cmd = \
            'singularity exec -B /explore/nobackup/projects/ilab,' + \
            '$NOBACKUP,/lscratch,/panfs/ccds02/nobackup/projects/ilab ' + \
            f'{self.conf.container_path} ' + \
            'mpirun -np 40 --oversubscribe ./geogrid.exe'

        # run geogrid command
        os.system(geodrid_cmd)

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
    def setup_wrf_config(self):
        environment = Environment(loader=FileSystemLoader("templates/"))
        template = environment.get_template("message.txt")
        return