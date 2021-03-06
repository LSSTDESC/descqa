import os
import importlib
import yaml
from .config import base_catalog_dir


__all__ = ['available_catalogs', 'get_catalog_config', 'get_available_catalogs', 'load_catalog']


def load_yaml(yaml_file):
    """
    Load *yaml_file*. Ruturn a dictionary.
    """
    with open(yaml_file) as f:
        config = yaml.safe_load(f)
    return config


def import_subclass(subclass, package=None, required_base_class=None):
    """
    Import and return a subclass.
    *subclass_path* must be in the form of 'module.subclass'.
    """
    subclass = getattr(importlib.import_module('.'+subclass, package), subclass)
    if required_base_class:
        assert issubclass(subclass, required_base_class), "Provided class is not a subclass of *required_base_class*"
    return subclass


def get_catalog_config(catalog):
    """
    get the config dict of *catalog*
    """
    return available_catalogs[catalog]


def get_available_configs(config_dir, register=None):
    """
    Return (or update) a dictionary *register* that contains all config files in *config_dir*.
    """
    if register is None:
        register = dict()

    for config_file in os.listdir(config_dir):
        if config_file.startswith('_') or not config_file.lower().endswith('.yaml'):
            continue

        name = os.path.splitext(config_file)[0]
        config = load_yaml(os.path.join(config_dir, config_file))
        config['base_catalog_dir'] = base_catalog_dir
        if 'fn' in config:
            config['fn'] = os.path.join(base_catalog_dir, config['fn'])
        register[name] = config

    return register


def get_available_catalogs(include_default_only=False): # pylint: disable=unused-argument
    """
    Return *available_catalogs* as a dictionary
    """
    return available_catalogs


def load_catalog_from_config_dict(catalog_config):
    """
    Load a galaxy catalog using a config dictionary.

    Parameters
    ----------
    catalog_config : dict
        a dictionary of config options

    Return
    ------
    galaxy_catalog : instance of a subclass of BaseGalaxyCatalog

    See also
    --------
    load_catalog()
    """
    return import_subclass(catalog_config['reader'],
                           __package__,
                           object)(**catalog_config)


def load_catalog(catalog_name, config_overwrite=None):
    """
    Load a galaxy catalog as specified in one of the yaml file in catalog_configs.

    Parameters
    ----------
    catalog_name : str
        name of the catalog (without '.yaml')
    config_overwrite : dict, optional
        a dictionary of config options to overwrite

    Return
    ------
    galaxy_catalog : instance of a subclass of BaseGalaxyCatalog
    """
    if catalog_name.lower().endswith('.yaml'):
        catalog_name = catalog_name[:-5]

    if catalog_name not in available_catalogs:
        raise KeyError("Catalog `{}` does not exist in the register. See `available_catalogs`.".format(catalog_name))

    config = available_catalogs[catalog_name]

    if config_overwrite:
        config = config.copy()
        config.update(config_overwrite)

    return load_catalog_from_config_dict(config)


available_catalogs = get_available_configs(os.path.join(os.path.dirname(__file__), 'configs'))
