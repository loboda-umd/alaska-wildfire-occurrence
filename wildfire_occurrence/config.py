from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Wildfire Occurrence data configuration class (embedded with OmegaConf).
    """

    # WRF Variables

    # Directory to store output files
    working_dir: str

    # WPS path
    wps_path: str

    # WRF path
    wrf_path: str

    # Multinode option
    multi_node: Optional[bool] = False

    # Container path
    container_path: Optional[str] = None

    # Container mounting directories
    container_mounts: Optional[list] = None

    # Dictionary to store WPS configuration file options
    wps_config: Optional[dict] = field(
        default_factory=lambda: {'interval_seconds': 10800})

    # Dictionary to store WRF configuration file options
    wrf_config: Optional[dict] = field(
        default_factory=lambda: {
            'interval_seconds': 10800, 'num_metgrid_levels': 27})

    # Output filename from WRF to extract variables from
    wrf_output_filename: Optional[str] = 'wrfout_d02_*_00:00:00'

    # List for posprocessing of variables
    wrf_output_variables: Optional[List[str]] = field(
        default_factory=lambda: [
            'CFTotal',
            'CFLow',
            'CFMed',
            'CFHigh',
            'DZ700_850',
            'GPZ500',
            'GPZ700',
            'GPZ750',
            'GPZ850',
            'Helicity',
            'LCL',
            'LFC',
            'MCAPE',
            'MCIN',
            'PLI',
            'PW',
            'RAINTotal',
            'RH2',
            'RH500',
            'RH700',
            'RH800',
            'RH850',
            'SHOW',
            'SLP',
            'TD2',
            'TD500',
            'TT',
            'T2',
            'T500',
            'T750',
            'T850',
            'W500',
            'WA500'
        ]
    )

    # Lightning Model Variables

    # data filenames regex to where WRF variables reside for training
    data_regex_list: Optional[List[str]] = field(
        default_factory=lambda: ['*.tif'])

    # data filenames regex to where WRF variables reside for validation
    validation_data_regex_list: Optional[List[str]] = field(
        default_factory=lambda: ['*.tif'])

    # label filenames regex to where ALDN data resides
    label_regex_list: Optional[List[str]] = field(
        default_factory=lambda: ['*.gpkg'])

    # aoi to select from, select between Alaska, Boreal, Tundra
    aoi: Optional[str] = 'Alaska'

    # geometry file with area of interest
    aoi_geometry_filename: Optional[str] = '*.gpkg'

    # directory to output data
    lightning_working_dir: Optional[str] = 'output'

    # output filename for lightning database
    lightning_output_filename: Optional[str] = 'my_database.gpkg'

    # output filename for model
    lightning_model_filename: Optional[str] = 'my_model.sav'

    # data dir for lightning model
    lightning_data_dir: Optional[str] = 'my_model'
