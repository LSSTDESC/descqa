# DESCQA web view

This directory hosts the cgi scripts for displaying the results.


## How to debug:

Debugging cgi scripts is a pain.

If you think the web interface is not working correctly, you should first try enabling `cgitb` in `index.cgi`. This allows you to see the traceback message on the web, and make debugging much easier. 

However, you may encouter "Internal Server Error", which is not very informative. There are some possibilities: 

1.  The script does not have the correct permission. run `./fix_permission.sh` to fix it.
2.  There's syntax errors in the script. In this case, run `python <script.py>` on the server to see what's wrong. 
3.  The http header is not printed correctly. 
