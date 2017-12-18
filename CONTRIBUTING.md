# How to contribute to DESCQA?

### Step 1: Fork the DESCQA GitHub repo and clone a local copy on NERSC

_Note_: You can skip this step and start with Step 2, if you already did it once.

Go to GitHub and [fork](https://guides.github.com/activities/forking/) the DESCQA GitHub repo.
Once you have a forked repo, make a local clone on NERSC:

    cd /your/own/directory
    git clone git@github.com:YourGitHubUsername/descqa.git
    cd descqa
    git remote add upstream https://github.com/LSSTDESC/descqa.git

_Note_: If you don't have a GitHub ssh key set up, replace `git@github.com:YourGitHubUsername/descqa.git` with `https://github.com/YourGitHubUsername/descqa.git` in the second line.


### Step 2: Sync the master branch of your fork repo with upstream

_Note_: Do *not* skip this step!

    cd /your/own/directory/descqa
    git checkout master
    git pull upstream master
    git push origin master


### Step 3: Create a new branch

    git checkout -b newBranchName


### Step 4: Develop

Make changes inside your local descqa clone. Validation tests can be found under the `descqa` directory. See [here](descqa/README.md) for more detailed instruction on how to create a new test.

_Note_: Please write [2-3 compatible](http://python-future.org/compatible_idioms.html) code.


### Step 5: Commit your change

When you are happy about your changes, you can commit them. First, make sure you are in your local descqa clone:

    cd /your/own/directory/descqa

and check current status of change:

    git status

"Stage" everything you want to commit and then commit:

    git add <files changed>
    git commit -m <short but meaningful message>


### Step 6: Test

See the "How to run DESCQA" section below.


### Step 7: Iterate

Repeat steps 3, 4, 5, 6 as necessary.


### Step 8: Push your changes and create a pull request

First, push your changes to GitHub

    git push origin newBranchName

Then go to https://github.com/LSSTDESC/descqa/ to create a pull request.



## How to run DESCQA?

_Note: You need to run DESCQA on a NERSC machine._

### Step 1: Clone DESCQA on NERSC

On NERSC,

    cd your/own/directory
    git clone https://github.com/LSSTDESC/descqa.git


### Step 2: Run the master script

Make sure you are in your local descqa clone on NERSC:

    cd your/own/directory/descqa

Then you can simply run `./run_master.sh`; however, there are many useful options to be aware of.


#### master script options

-  You can add `-v` to  allow the error messages to be printed out, which is useful for debugging.

-  If you want to run only a subset of catalogs or tests, you can specify `--catalogs-to-run` (or `-c` for short) and `--validations-to-run` (or `-t` for short)

       ./run_master.sh -v -c CATALOG1 CATALOG2 -t TEST1 TEST2


-  You can also use wildcard, for example:

       ./run_master.sh -v -c *_test catalog3

   will run all available tests on all catalogs whose name ends in "_test" and also "catalog3", in verbose mode.

-  If you just want to see what catalogs and validation tests are available, you can run:

       ./run_master.sh -l

   which will print the names of all available catalogs in `GCRCatalogs` and all validation tests in `descqa`.

-  If you are working on `GCRCatalogs` and want to use DESCQA to test your own local version of `GCRCatalogs`, you can do:

       ./run_master.sh -p /path/to/dir/that/contains/GCRCatalogs <and other options as needed>

   Note the path here should be the directory that **contains** `GCRCatalogs`, *not* the `GCRCatalogs` directory itself.


### Step 3: Check results

As the master script is running, all the error messages will be printed out in real time if you have set `-v`. You can also go to the web interface to check you result:

https://portal.nersc.gov/project/lsst/descqa/v2/?run=all

