# How to contribute to DESCQA?

## Our GitHub workflow

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

_Note_: **Always do this!**

    cd /your/own/directory/descqa
    git checkout master
    git pull upstream master
    git push origin master


### Step 3: Create a new branch

    git checkout -b newBranchName master


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


## Integration test

We use [GitHub Actions](https://help.github.com/en/actions/automating-your-workflow-with-github-actions)
to run integration tests on pull requests (PRs).
Once you submit a PR, you will see whether your PR passes the integration test.

Because it's difficult to directly test most of the DESCQA functionalities
(as they need to be run on NERSC), we heavily rely on linters for integration
tests. In particular we use flake8 and pylint to check any potential syntax or
logic errors. PRs need to pass these tests before they can be merged.

You can see our integration test workflow [here](.github/workflows/pythonpackage.yml).

### How to fix the errors?

Check the output of GitHub Actions and it'll tell you what's wrong.
You can then fix the code accordingly.

Note that linters are not perfect, sometimes they report false positive.
Other times the code runs fine but has [anti-patterns](https://en.wikipedia.org/wiki/Anti-pattern)
which trigger the linters.
You should change your code to follow best practices and make linter happy.
In cases where you are certain that the linter made a false positive report,
you can add the follow comment in your code to disable specific errors.
You should add these exceptions at where the error takes place.

```python
# pylint: disable=<pylint code>  # noqa: <flake8 code>
```

### Build failed due to dependencies

If your test depends on additional python packages or other libraries,
first you should ask yourself if these additional dependencies are _really_ needed.
Even if we manage to include these dependencies in the integration test, we still
need to add them to our NERSC environment, which may be difficult.

If you need to add additional python packages as dependencies, they should be added to
`setup.py`, under `extras_require` > `full`. The integration test will automatically
install all the packages specified there.

If you need to add non-python libraries as dependencies, please do make sure you
_really_ need them. Discuss with other DESCQA developers.
There's not simple intructions for how to add non-python libraries.


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

https://portal.nersc.gov/projecta/lsst/descqa/v2/?run=all

