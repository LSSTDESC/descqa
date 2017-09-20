# DESCQA

**The master branch now hosts DESCQA2 (which is still under development). If you are looking for DESCQA1, go to [v1 branch](https://github.com/LSSTDESC/descqa/tree/v1)**

Starting from DESCQA2, the access of catalogs (including the readers) are moved to [GCRCatalogs](https://github.com/LSSTDESC/gcr-catalogs)

This repository contains the descqa simulation validation framework derived from FlashTest/FlashTestView. It will be used initially to do automated testing of simulated galaxy catalogs but may also expand later to encompass other types of simulation validation.


## Instruction for DESCQA developers

_Note: Do Steps 1 through 6 under one of your own directory on a NERSC machine. Do **not** go to the main descqa directory (unless you made changes to the web interface, in which case, see Step 7)._


### Step 1: Clone or pull the git repo

If you have never cloned the repo before, run:

    cd your/own/directory
    git clone https://github.com/LSSTDESC/descqa.git descqa-local

Or, if you have already cloned the repo, then run:

    cd your/own/directory/descqa-local
    git pull

_Note: If you have been using the `hackweek16` branch, please rename it to `master` by running:_

    cd your/own/directory/descqa-local
    git branch -m hackweek16 master
    git pull origin master
    git config --local branch.master.merge refs/heads/master    


### Step 2: Develop

Hack on! Make changes inside your local descqa clone. 


### Step 3: Test

Make sure you are in your local descqa clone:

    cd your/own/directory/descqa-local

And simply run:

    ./run_master.sh -v

The `-v` argument allows the error messages to be printed out, which is useful for debugging. 

If you want to run only a subset of catalogs or tests, you can specify `--catalogs-to-run` (or `--rc` for short) and `--validations-to-run` (or `--rv` for short) 
    
    ./run_master.sh -v --rc CATALOG1 CATALOG2 --rv TEST1 TEST2


_Note: If you see error message about conflicting modules, please unload the module in conflict and then try again. To unload a module, run:_

    module unload <module_name>


### Step 4: Check results

As the master script is running, all the error messages will be printed out in real time if you have set `-v`. You can also go to the web interface to check you result:

https://portal.nersc.gov/project/lsst/descqa/index.cgi?run=all


### Step 5: Iterate

Repeat steps 2, 3, 4 as necessary.


### Step 6: Commit and push your changes (only if you are a DESCQA member)

Now that you are happy about your changes, you can commit them. First, make sure you are in your local descqa clone:

    cd your/own/directory/descqa-local

and check current status of change:

    git status

"Stage" everything you want to commit and then commit: 

    git add <files to stage> 
    git commit -m <short but meaningful message>
    
For now we do *not* use pull requests for changes, so unless you have made significant changes (in which case, contact the QA team frist), do a pull rebase first (do **not** skip this step!):

    git pull -r
    
Hopefullly there's no conflict. If there are, resolve them, and commit again. Once you're done, push your changes back:

    git push
    
    
### Step 7: update main descqa direcotory (ONLY IF you made changes to the web interface)

_Note: you don't need to do this step **unless** you made changes to the web interface._

Go to the main descqa direcotory on NERSC and pull changes from github:

    git pull
    
And fix permissions:

    cd www
    ./fix_permission
   

## Code structure

- `master.py`: the master script to start a test run
- `run_master.sh`: a convenient shell script to set enviornment variables/paths before running `master.py`
- `config_catalog.py`: config file to set up catalogs (to specify catalog files and readers)
- `config_validation.py`: config file to set up validation tests (to specify test classes and arguments)
- `archiver.py`: to clean up (archive) the output directory
- `reader/`: directory that hosts all the reader classes
- `validation_code/`: directory that hosts all the validation test classes and relevent utilities
- `validation_data/`: directory that hosts small data files that validation tests need
- `www/`: directory that hosts the web interface

_Note: actual catalog files are not in this repo as they are generally much bigger._

