# DESCQA web view

this directory hosts the cgi scripts for displaying the results.


## Code structure

- "Big table" page: `home.py`, which also calls `utils/invocations_simple.py` to generate the content and cache under `cache`. Use `config` to set options

- "Matrix" page: `viewer/viewBuilds.py`

- "Show plots" page: `viewer/viewBuild.py` is the main frame. On the left, `viewer/leftFrame.py` uses `viewer/viewBuildTemplate.ezt` as a template to generate the comparison. On the right it shows the text or png file with `viewer/viewFile.py`.

