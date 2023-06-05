import os
import logging
import datetime
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
        self.output_dir = os.path.join(
            self.conf.working_dir,
            f'{self.start_date.strftime("%Y-%m-%d")}_' +
            f'{self.start_date.strftime("%Y-%m-%d")}'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f'Created output directory {self.output_dir}')

        # Setup data_dir
        self.data_dir = os.path.join(self.output_dir, 'data')

    # -------------------------------------------------------------------------
    # download
    # -------------------------------------------------------------------------
    def download(self):

        # Working on the setup of the project
        logging.info('Starting download pipeline step')

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

        return

    # -------------------------------------------------------------------------
    # download
    # -------------------------------------------------------------------------
    def geogrid(self):
        logging.info('Running geogrid')
        return
