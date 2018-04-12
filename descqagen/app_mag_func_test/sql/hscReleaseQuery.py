"""
python module to query HSC CAS server

This is heavily modified code borrowed from: https://hsc-gitlab.mtk.nao.ac.jp/snippets/17
"""

import json
import urllib2
import time
import sys

#store query server information
version = 20170216.1
release_version = 'pdr1'
api_url = 'https://hsc-release.mtk.nao.ac.jp/datasearch/api/catalog_jobs/'


__all__ = ['submitJob', 'deleteJob', 'jobStatus', 'jobCancel', 'download']
__author__ = ['Duncan Campbell']


def main():
    """
    Run SQL qury on HSC CAS database.

    This script can be run in the terminal as:
        user$ hsc_query.py test.sql test_out.sql
    """

    # Read in the user's creditionals.
    user_info_file = './hsc_credentials.txt'
    f = open(user_info_file, 'r')
    username = f.readline().strip()
    password = f.readline().strip()
    f.close()
    if (username == 'username') | (password == 'password'):
        print("Please alter the file: {0}, to be your username and password".format(user_info_file))

    credential = {'account_name': username, 'password': password}

    if len(sys.argv) == 3:
        sql_file = sys.argv[1]
        savename = sys.argv[2]
    else:
        print("main takes two positional arguments: path/to/file/containing/sql/query and output file.")

    # Read in the sql query string
    f = open(sql_file, 'r')
    sql = f.read()
    f.close()

    # Submit the job
    job = submitJob(sql, credential)
    print('running job: ', job['id'])

    #block the script until the job is done running
    blockUntilJobFinishes(credential, job['id'])

    out_file = open(savename, "w")
    download(credential, job['id'], out_file)

    #delete job after downloading
    deleteJob(credential, job['id'])



def httpJsonPost(url, data):
    """
    """
    data['clientVersion'] = version
    postData = json.dumps(data)
    return httpPost(url, postData, {'Content-type': 'application/json'})


def httpPost(url, postData, headers):
    """
    """
    req = urllib2.Request(url, postData, headers)
    res = urllib2.urlopen(req)
    return res


class QueryError(Exception):
    pass


def submitJob(sql, credential, out_format='csv.gz'):
    """
    Submit a job to the HSC CAS server.

    Paramaters
    ==========
    credential : dict
        dictionary contraing username and password

    sql : string
        sql query

    out_format : string
        file format

    Returns
    =======
    job
    """
    url = api_url + 'submit'
    catalog_job = {
        'sql'                     : sql,
        'out_format'              : out_format,
        'include_metainfo_to_body': True,
        'release_version'         : release_version,
    }
    postData = {'credential': credential, 'catalog_job': catalog_job, 'nomail': True, 'skip_syntax_check': True}
    res = httpJsonPost(url, postData)
    job = json.load(res)
    return job


def jobStatus(credential, job_id):
    """
    get status of job
    """
    url = api_url + 'status'
    postData = {'credential': credential, 'id': job_id}
    res = httpJsonPost(url, postData)
    job = json.load(res)
    return job


def jobCancel(credential, job_id):
    """
    cancel job
    """
    url = api_url + 'cancel'
    postData = {'credential': credential, 'id': job_id}
    httpJsonPost(url, postData)


def blockUntilJobFinishes(credential, job_id, max_time=60.0*60.0):
    """
    run until job finishes
    """

    interval = 10.0 # wait 10 seconds before checking again

    start = time.time()
    while True:
        time.sleep(interval)

        job = jobStatus(credential, job_id)
        print(job)
        if job['status'] == 'error':
            raise QueryError, 'query error: ' + job['error']
        if job['status'] == 'done':
            break

        dt = time.time()-start

        if dt>max_time:
            print("job time maxed out.")
            jobCancel(credential, job_id)
            deleteJob(credential, job_id)
            sys.exit()


def download(credential, job_id, out):
    """
    download the reuslt of a job and save to output file
    """
    url = api_url + 'download'
    postData = {'credential': credential, 'id': job_id}
    res = httpJsonPost(url, postData)
    bufSize = 64 * 1<<10 # 64k
    while True:
        buf = res.read(bufSize)
        out.write(buf)
        if len(buf) < bufSize:
            break


def deleteJob(credential, job_id):
    """
    delete a jopb from the CAS server
    """
    url = api_url + 'delete'
    postData = {'credential': credential, 'id': job_id}
    httpJsonPost(url, postData)



if __name__ == '__main__':
    main()
