import sys
import pathlib
import configparser
import os

try:
    from xdg import BaseDirectory
except ImportError:
    BaseDirectory = None

PROGRAMNAME = 'pter_analysis'
HERE = pathlib.Path(os.path.abspath(__file__)).parent
HOME = pathlib.Path.home()
CONFIGDIR = HOME / ".config" / PROGRAMNAME
CONFIGFILE = HOME / ".config" / PROGRAMNAME / (PROGRAMNAME + ".conf")

if BaseDirectory is not None:
    CONFIGDIR = pathlib.Path(BaseDirectory.save_config_path(PROGRAMNAME) or CONFIGDIR)
    CONFIGFILE = CONFIGDIR / (PROGRAMNAME + ".conf")

DEFAULT_CONFIG = {
    'General': {
        'work-day-length': '8h',
        'work-week-length': '5d',
        'todotxt-file': '',
        'matplotlib-style': 'Solarize_Light2',
        'usetex': 'True',
        'search-case-sensitive': 'True',
    },
}

def get_config(pth):
    config = configparser.ConfigParser(interpolation=None)
    config.read_dict(DEFAULT_CONFIG)
    conffile = pathlib.Path(os.path.abspath(os.path.expanduser(pth)))

    if conffile.exists() and conffile.is_file():
        config.read([conffile])

    return config