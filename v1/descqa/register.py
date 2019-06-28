import os
import importlib
import yaml
from .base import BaseValidationTest


__all__ = ['available_validations', 'load_validation', 'load_validation_from_config_dict']


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
    """
    subclass = getattr(importlib.import_module('.'+subclass, package), subclass)
    if required_base_class:
        assert issubclass(subclass, required_base_class), "Provided class is not a subclass of *required_base_class*"
    return subclass


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
        config['test_name'] = name
        config['base_data_dir'] = os.path.join(os.path.dirname(__file__), 'data')
        register[name] = config

    return register


def load_validation_from_config_dict(validation_config):
    """
    Load a validation test using a config dictionary.

    Parameters
    ----------
    validation_config : dict
        a dictionary of config options

    Return
    ------
    validation_test : instance of a subclass of BaseValidationTest

    See also
    --------
    load_catalog()
    """
    return import_subclass(validation_config['module'],
                           __package__,
                           BaseValidationTest)(**validation_config)


def load_validation(validation_name, config_overwrite=None):
    """
    Load a validation test as specified in one of the yaml file in configs.

    Parameters
    ----------
    validation_name : str
        name of the validation test (without '.yaml')
    config_overwrite : dict, optional
        a dictionary of config options to overwrite

    Return
    ------
    validation_test : instance of a subclass of BaseValidationTest
    """
    if validation_name.lower().endswith('.yaml'):
        validation_name = validation_name[:-5]

    if validation_name not in available_validations:
        raise KeyError("Validation `{}` does not exist in the register. See `available_validations`.".format(validation_name))

    config = available_validations[validation_name]

    if config_overwrite:
        config = config.copy()
        config.update(config_overwrite)

    return load_validation_from_config_dict(config)


available_validations = get_available_configs(os.path.join(os.path.dirname(__file__), 'configs'))
