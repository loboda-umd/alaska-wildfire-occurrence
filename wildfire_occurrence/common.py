import sys
import argparse
import omegaconf
from glob import glob
from datetime import datetime
from omegaconf.listconfig import ListConfig
from wildfire_occurrence.config import Config


# -------------------------------------------------------------------------
# read_config
# -------------------------------------------------------------------------
def read_config(filename: str, config_class=Config):
    """
    Read configuration filename and initiate objects
    """
    # Configuration file initialization
    schema = omegaconf.OmegaConf.structured(config_class)
    conf = omegaconf.OmegaConf.load(filename)
    try:
        conf = omegaconf.OmegaConf.merge(schema, conf)
    except BaseException as err:
        sys.exit(f"ERROR: {err}")
    return conf


# -------------------------------------------------------------------------
# validate_date
# -------------------------------------------------------------------------
def valid_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "not a valid date: {0!r}".format(s)
        raise argparse.ArgumentTypeError(msg)


# -------------------------------------------------------------------------
# get_filenames
# -------------------------------------------------------------------------
def get_filenames(data_regex: str) -> list:
    """
    Get filename from list of regexes
    """
    # get the paths/filenames of the regex
    filenames = []
    if isinstance(data_regex, list) or isinstance(data_regex, ListConfig):
        for regex in data_regex:
            filenames.extend(glob(regex))
    else:
        filenames = glob(data_regex)
    assert len(filenames) > 0, f'No files under {data_regex}'
    return sorted(filenames)
