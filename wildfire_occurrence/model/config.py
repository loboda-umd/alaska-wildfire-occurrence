from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Wildfire Occurrence data configuration class (embedded with OmegaConf).
    """

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
