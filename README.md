# DESCQA

This repository contains the DESCQA simulation validation framework derived from FlashTest/FlashTestView. It will be used initially to do automated testing of simulated galaxy catalogs but may also expand later to encompass other types of simulation validation.

**The master branch now hosts DESCQA2 (which is still under development). If you are looking for DESCQA1, go to [v1 branch](https://github.com/LSSTDESC/descqa/tree/v1)**

**Starting from DESCQA2, the configurations and readers of catalogs are separated from DESCQA and are now hosted in [GCRCatalogs](https://github.com/LSSTDESC/gcr-catalogs).**


## Instruction for DESCQA developers

_Note: You can do Steps 1 through 6 under one of your own directory on a NERSC machine._

### Step 0: Activate DESCQA python enviornment

On NERSC machine, change to bash or zsh and run :

    source /global/common/cori/contrib/lsst/apps/anaconda/4.4.0-py2/bin/activate
    source activate DESCQA


### Step 1: Clone or pull the git repo

If you have never cloned the repo before, run:

    cd your/own/directory
    git clone git@github.com:LSSTDESC/descqa.git

If you don't have GitHub ssh key set up, you can do
    
    git clone https://github.com/LSSTDESC/descqa.git

Or, if you have already cloned the repo, then run:

    cd your/own/directory/descqa
    git pull


### Step 2: Create a new branch

    cd your/own/directory/descqa
    git checkout -b newBranchName


### Step 3: Develop

Hack on! Make changes inside your local descqa clone. See [here](https://github.com/LSSTDESC/descqa/blob/master/validation_code/README.md) for more detailed instruction on how to create a new test.


### Step 4: Test

Make sure you are in your local descqa clone:

    cd your/own/directory/descqa

And simply run:

    ./run_master.sh -v

The `-v` argument allows the error messages to be printed out, which is useful for debugging. 

If you want to run only a subset of catalogs or tests, you can specify `--catalogs-to-run` (or `-c` for short) and `--validations-to-run` (or `-t` for short) 
    
    ./run_master.sh -v -c CATALOG1 CATALOG2 -t TEST1 TEST2


### Step 5: Check results

As the master script is running, all the error messages will be printed out in real time if you have set `-v`. You can also go to the web interface to check you result:

https://portal.nersc.gov/project/lsst/descqa/v2/www/index.cgi?run=all


### Step 6: Commit your change

Whenyou are happy about your changes, you can commit them. First, make sure you are in your local descqa clone:

    cd your/own/directory/descqa

and check current status of change:

    git status

"Stage" everything you want to commit and then commit: 

    git add <files changed>
    git commit -m <short but meaningful message>


### Step 7: Iterate

Repeat steps 3, 4, 5, 6 as necessary.


### Step 8: Push your changes and create a pull request

First, push your changes to GitHub
    
    git push origin newBranchName
    
Then go to https://github.com/LSSTDESC/descqa/ to create a pull request. 
    
    
### Step 99: update main descqa direcotory (ONLY IF you made changes to the web interface)

_Note: you don't need to do this step **unless** you made changes to the web interface._

Go to the main descqa direcotory on NERSC and pull changes from github:

    git pull
    
And fix permissions:

    cd www
    ./fix_permission
   

## Code structure

- `master.py`: the master script to start a test run
- `run_master.sh`: a convenient shell script to set enviornment variables/paths before running `master.py`
- `archiver.py`: to clean up (archive) the output directory
- `validation_configs/`: directory that hosts all validation test config YAML files
- `validation_code/`: directory that hosts all the validation test classes and relevent utilities
- `validation_data/`: directory that hosts small data files that validation tests need
- `www/`: directory that hosts the web interface

_Note: actual catalog files are not in this repo as they are generally much bigger._


