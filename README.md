# DESCQA

This repository contains the descqa simulation validation framework derived from FlashTest/FlashTestView. It will be used initially to do automated testing of simulated galaxy catalogs but may also expand later to encompass other types of simulation validation.


## Usage

    ./run_master.sh

use `-h` to see help


## Code structure

- `master.py`: the master script to start a test run
- `run_master.sh`: a convenient shell script to set some enviornment variables before running `master.py`
- `config_catalog.py`: config file to set up catalogs
- `config_validation.py`: config file to set up validation tests
- `archiver.py`: to clean up (archive) the output directory
 
- `reader`: directory to host all the reader classes
- `validation_code`: directory to host all the validation test classes and relevent utilities
- `validation_data`: directory to host small data files that validation tests need

note: actual catalogs are not in this repo as they are generally much bigger.
