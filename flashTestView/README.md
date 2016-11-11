# DESCQA web view

this directory hosts the cgi scripts for displaying the results.


## Code structure

- "Big table page": `home.py`, which also calls `lib/invocations_simple.py` to generate the content.

- "Build page": `viewer/viewBuilds.py` which uses `viewer/viewBuildsTemplate.ezt` as a template to generate the page.

- "Comparison page": `viewer/viewBuild.py` is the frame page. On the left, `viewer/leftFrame.py` uses `viewer/viewBuildTemplate.ezt` as a template to generate the comparison. On the right it shows the text or png file with `viewer/view*File.py`.

