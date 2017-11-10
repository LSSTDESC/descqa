# DESCQA web view

This directory hosts the cgi scripts for displaying the results.


## How to debug:

Debugging cgi scripts is a pain. Now, all python scripts already have `cgitb` enabled, which allows you to see the traceback message on the web, and make debugging much easier. 

However, you may encouter "Internal Server Error", which is not very informative. This usually means two possibilities: 

1.  The script does not have the correct permission. run `./fix_permission.sh` to fix it.
2.  There's syntax errors in the script. In this case, run `python <script.py>` on the server to see what's wrong. 
