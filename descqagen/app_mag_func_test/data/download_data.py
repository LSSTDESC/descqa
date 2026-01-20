"""
run sql queries on HSC database
"""

from __future__ import print_function, division
import sys
sys.path.append("..")
from defaults import PROJECT_DIRECTORY, DATA_DIRECTORY


def main():
    """
    run scripted SQL download.

    to download the HSC data for this project, in the temrinal window type:
        $user python download_data.py data

    to download the HSC randoms for this project, in the temrinal window type:
        $user python download_data.py randoms

    Before running this script, edit the hsc_credentials.txt file in the sql directory.
    """

    if sys.argv[1] == 'data':
        sql_file = PROJECT_DIRECTORY + 'sql/hsc_data.sql'
        savename = DATA_DIRECTORY + 'hsc_data.csv'
    elif sys.argv[1] == 'randoms':
        sql_file = PROJECT_DIRECTORY + 'sql/hsc_randoms.sql'
        savename = DATA_DIRECTORY + 'hsc_randoms.csv'
    else:
        print("specify either 'data' or 'randoms' as a positional argument.")
        sys.exit()

    # read in username and password for HSC account
    user_info_file = PROJECT_DIRECTORY + 'sql/hsc_credentials.txt'
    f = open(user_info_file, 'r')
    username = f.readline().strip()
    password = f.readline().strip()
    f.close()
    if (username == 'username') | (password == 'password'):
        print("Please alter the file: {0}, to be your username and password".format(user_info_file))
        print("It is possible your username and password are 'username' and 'password'. Shame on you.")

    # open sql query string
    f = open(sql_file, 'r')
    sql = f.read()
    f.close()

    # run query script
    import subprocess
    exec_script = PROJECT_DIRECTORY+'sql/hscReleaseQuery.py'
    process = subprocess.call(['python', exec_script, '--user', username, '--password', password, sql_file])



if __name__ == "__main__":
    main()
