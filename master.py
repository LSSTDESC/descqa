import os
import sys
import shutil
import time
import json
import logging
import traceback
import StringIO
import importlib
import argparse
import collections
import fnmatch

pjoin = os.path.join


class ExceptionAndStdStreamCatcher():
    def __init__(self):
        self.output = ''
        self.has_exception = False


class CatchExceptionAndStdStream():
    def __init__(self, catcher):
        self._catcher = catcher
        self._stream = StringIO.StringIO()

    def __enter__(self):
        self._stdout = sys.stdout
        self._stdout.flush()
        sys.stdout = self._stream
        self._stderr = sys.stderr
        self._stderr.flush()
        sys.stderr = self._stream

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._stream.flush()
        if exc_type:
            traceback.print_exception(exc_type, exc_value, exc_tb, file=self._stream)
            self._catcher.has_exception = True
        self._catcher.output = self._stream.getvalue()
        self._stream.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        return True


def quick_import(module_name):
    return getattr(importlib.import_module(module_name), module_name)


def process_config(config_dict, keys_wanted=None):
    d = {k: config_dict[k] for k in config_dict if not k.startswith('_')}
    if keys_wanted is None:
        return d

    keys_wanted = set(keys_wanted)
    keys_available = d.keys()
    keys_to_return = set()

    for k in keys_wanted:
        keys = fnmatch.filter(keys_available, k)
        if not keys:
            raise ValueError("{} does not present in config ({})...".format(k, ', '.join(keys_available)))
        map(keys_to_return.add, keys)

    return {k: d[k] for k in d if k in keys_to_return}


def create_logger(verbose=False):
    log = logging.getLogger()
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    logFormatter = logging.Formatter('[%(levelname)-5.5s][%(asctime)s] %(message)s')
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    log.addHandler(consoleHandler)
    return log


def check_copy(src, dst):
    if os.path.exists(dst):
        raise OSError('{} already exists'.format(dst))
    if os.path.isdir(src):
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('.*', '*~', '#*'))
    elif os.path.isfile(src):
        shutil.copy(src, dst)
    else:
        raise OSError('{} does not exist'.format(src))
    return dst


def make_output_dir(root_output_dir, create_subdir=True):
    if create_subdir:
        if not os.path.isdir(root_output_dir):
            raise OSError('{} does not exist'.format(root_output_dir))
        new_dir_name = time.strftime('%Y-%m-%d')
        output_dir = pjoin(root_output_dir, new_dir_name)
        if os.path.exists(output_dir):
            i = max((int(s.partition('_')[-1] or 0) for s in os.listdir(root_output_dir) if s.startswith(new_dir_name)))
            output_dir += '_{}'.format(i+1)
    else:
        if os.path.exists(root_output_dir):
            raise OSError('{} already exists'.format(root_output_dir))
        output_dir = root_output_dir
    os.mkdir(output_dir)
    return output_dir


class TaskDirectory():
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self._path = {}
        self._status = collections.defaultdict(dict)

    def get_path(self, validation_name, catalog_name=None):
        key = (validation_name, catalog_name)
        if key not in self._path:
            self._path[key] = pjoin(self.output_dir, validation_name, catalog_name) if catalog_name else pjoin(self.output_dir, validation_name)
            if not os.path.exists(self._path[key]):
                os.makedirs(self._path[key])
        return self._path[key]

    def set_status(self, validation_name, catalog_name, status, summary=None):
        status = status.strip().upper()
        if not any(status.endswith(t) for t in ('PASSED', 'FAILED', 'ERROR', 'SKIPPED')):
            raise ValueError('status message not set correctly!')
        
        self._status[validation_name][catalog_name] = status
        with open(pjoin(self.get_path(validation_name, catalog_name), 'STATUS'), 'w') as f:
            f.write(status + '\n')
            if summary:
                f.write(summary.strip() + '\n')

    def get_status(self, validation_name=None, catalog_name=None):
        if catalog_name:
            return self._status[validation_name][catalog_name]
        elif validation_name:
            return self._status[validation_name]
        else:
            return self._status


def run(tasks, validations_to_run, catalogs_to_run, log):
    validation_instance_cache = {} # to cache valiation instance

    def write_to_traceback(msg):
        with open(pjoin(final_output_dir, 'traceback.log'), 'a') as f:
            f.write(msg)
    
    # loading catalog usually takes longer, so we put catalogs_to_run in outer loop
    for catalog_name, catalog in catalogs_to_run.iteritems():
        # check if there's still valid validations to run
        if all(isinstance(validation_instance_cache.get(v), basestring) for v in validations_to_run):
            log.debug('skipping "{}" catalog as there are errors in all validations'.format(catalog_name))
            continue

        # try loading the catalog
        log.info('loading "{}" catalog...'.format(catalog_name))
        catcher = ExceptionAndStdStreamCatcher()
        with CatchExceptionAndStdStream(catcher):
            Reader = quick_import(catalog.reader)
            gc = Reader(**catalog.kwargs)
        if catcher.has_exception:
            log.error('error occured when loading "{}" catalog...'.format(catalog_name))
            log.debug('stdout/stderr and traceback:\n' + catcher.output)
            gc = catcher.output
        elif catcher.output:
            log.debug('stdout/stderr while loading "{}" catalog:\n'.format(catalog_name) + catcher.output)

        # loop over validations_to_run
        for validation_name, validation in validations_to_run.iteritems():
            # get the final output path, set test kwargs
            final_output_dir = tasks.get_path(validation_name, catalog_name)
            validation.kwargs['test_name'] = validation_name

            # if gc is an error message, log it and abort
            if isinstance(gc, basestring):
                write_to_traceback(gc)
                tasks.set_status(validation_name, catalog_name, 'LOAD_CATALOG_ERROR')
                continue

            # try loading ValidationTest class/instance
            if validation_name not in validation_instance_cache:
                catcher = ExceptionAndStdStreamCatcher()
                with CatchExceptionAndStdStream(catcher):
                    ValidationTest = quick_import(validation.module)
                    vt = ValidationTest(**validation.kwargs)
                if catcher.has_exception:
                    log.error('error occured when preparing "{}" test'.format(validation_name))
                    log.debug('stdout/stderr and traceback:\n' + catcher.output)
                    vt = catcher.output
                elif catcher.output:
                    log.debug('stdout/stderr while preparing "{}" test:\n'.format(validation_name) + catcher.output)

                # cache the ValidationTest instance for future use
                validation_instance_cache[validation_name] = vt
            
            else:
                vt = validation_instance_cache[validation_name]
            
            #if vt is an error message, log it and abort
            if isinstance(vt, basestring): 
                write_to_traceback(vt)
                tasks.set_status(validation_name, catalog_name, 'VALIDATION_TEST_MODULE_ERROR')
                continue

            # run validation test
            catcher = ExceptionAndStdStreamCatcher()
            with CatchExceptionAndStdStream(catcher):
                result = vt.run_validation_test(gc, catalog_name, final_output_dir)
                assert result.status in ('PASSED', 'FAILED', 'SKIPPED') and isinstance(result.summary, basestring)

            if catcher.output:
                write_to_traceback(catcher.output)
                if catcher.has_exception:
                    log.error('error occured when running "{}" test on "{}" catalog...'.format(validation_name, catalog_name))
                    log.debug('stdout/stderr and traceback:\n' + catcher.output)
                    tasks.set_status(validation_name, catalog_name, 'RUN_VALIDATION_TEST_ERROR')
                    continue
                else:
                    log.debug('stdout/stderr while running "{}" test on "{}" catalog:\n'.format(validation_name, catalog_name) + catcher.output)
                
            tasks.set_status(validation_name, catalog_name, 'VALIDATION_TEST_{}'.format(result.status), result.summary)
            log.info('{} "{}" test on "{}" catalog'.format('skipping' if result.status == 'SKIPPED' else 'finishing', 
                    validation_name, catalog_name))



def call_summary_plot(tasks, validations_to_run, log):
    for validation in validations_to_run:
        catalog_list = []
        for catalog, status in tasks.get_status(validation).iteritems():
            if status.endswith('PASSED') or status.endswith('FAILED'):
                catalog_list.append((catalog, tasks.get_path(validation, catalog)))
        
        if not catalog_list:
            continue

        module_name = validations_to_run[validation].module

        catcher = ExceptionAndStdStreamCatcher()
        with CatchExceptionAndStdStream(catcher):
            plot_summary_func = getattr(importlib.import_module(module_name), 'plot_summary')
            plot_summary_func(pjoin(tasks.get_path(validation), 'summary_plot.png'), catalog_list, validations_to_run[validation].kwargs)

        if catcher.has_exception:
            log.error('error occured when generating summary plot for "{}" test...'.format(validation))
            log.debug('stdout/stderr and traceback:\n' + catcher.output)
        elif catcher.output:
            log.debug('stdout/stderr while generating summary plot for "{}" test:\n'.format(validation) + catcher.output)

        if catcher.output:
            with open(pjoin(tasks.get_path(validation), 'summary_plot.log'), 'w') as f:
                f.write(catcher.output)

            
def get_status_report(tasks):

    report = StringIO.StringIO()
    status = tasks.get_status()

    for validation in status:
        report.write('-'*50 + '\n')
        report.write(validation + '\n')
        report.write('-'*50 + '\n')
        l = max(len(catalog) for catalog in status[validation])
        l += 3
        
        for catalog in status[validation]:
            s = status[validation][catalog]
            report.write('{{:{}}}{{}}\n'.format(l).format(catalog, s))

    report.write('-'*50 + '\n')
    
    report_content = report.getvalue()
    report.close()

    return report_content


def write_master_status(master_status, tasks, output_dir):
    master_status['end_time'] = time.time()
    master_status['status_count'] = {}
    master_status['status_count_group_by_catalog'] = {}

    status = tasks.get_status()
    status_by_catalog = collections.defaultdict(list)
    for validation in status:
        master_status['status_count'][validation] = collections.Counter(status[validation].values())
        for catalog in status[validation]:
            status_by_catalog[catalog].append(status[validation][catalog])

    for catalog in status_by_catalog:
        master_status['status_count_group_by_catalog'][catalog] = collections.Counter(status_by_catalog[catalog])

    with open(pjoin(output_dir, 'STATUS.json'), 'w') as f:
        json.dump(master_status, f, indent=True)


def group_by_catalog(tasks, catalogs_to_run, validations_to_run, output_dir):
    catalog_group_dir = os.path.join(output_dir, '_group_by_catalog')
    os.mkdir(catalog_group_dir)
    for catalog in catalogs_to_run:
        this_catalog_dir = os.path.join(catalog_group_dir, catalog)
        os.mkdir(this_catalog_dir)
        for validation in validations_to_run:
            os.symlink(tasks.get_path(validation, catalog), os.path.join(this_catalog_dir, validation))


def get_username():
    for k in ('LOGNAME', 'USER', 'LNAME', 'USERNAME'):
        user = os.getenv(k)
        if user:
            return user
    return 'UNKNOWN'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_output_dir')
    parser.add_argument('--no-subdir', dest='subdir', action='store_false', help='if set, no sub-directory will be created, and in this case, `root_output_dir` must not yet exist')
    parser.add_argument('--validation-config', default='config_validation.py', help='validation config file')
    parser.add_argument('--catalog-config', default='config_catalog.py', help='catalog config file')
    parser.add_argument('--validations-to-run', metavar='VALIDATION', nargs='+', help='to run only a subset of validations')
    parser.add_argument('--catalogs-to-run', metavar='CATALOG', nargs='+', help='to run only a subset of catalogs')
    parser.add_argument('-v', '--verbose', action='store_true', help='to display all debug messages')
    parser.add_argument('-m', '--comment', help='to attact a comment to this run')
    args = parser.parse_args()

    master_status = {}
    master_status['user'] = get_username()
    master_status['start_time'] = time.time()
    if args.comment:
        master_status['comment'] = args.comment

    log = create_logger(verbose=args.verbose)

    log.debug('creating output directory...')
    output_dir = make_output_dir(args.root_output_dir, args.subdir)
    open(pjoin(output_dir, '.lock'), 'w').close()
    try: # we want to remove ".lock" file even if anything went wrong
        snapshot_dir = pjoin(output_dir, '_snapshot')
        os.mkdir(snapshot_dir)
        log.info('output of this run is stored in {}'.format(output_dir))

        log.debug('copying config files...')
        check_copy(args.validation_config, pjoin(snapshot_dir, 'validation_config.py'))
        check_copy(args.catalog_config, pjoin(snapshot_dir, 'catalog_config.py'))

        log.debug('importing config files...')
        sys.path.insert(0, snapshot_dir)
        import validation_config as vc
        import catalog_config as cc
        del sys.path[0]

        log.debug('processing config files...')
        validations_to_run = process_config(vc.__dict__, args.validations_to_run)
        catalogs_to_run = process_config(cc.__dict__, args.catalogs_to_run)
        if not validations_to_run or not catalogs_to_run:
            raise ValueError('not thing to run...')

        log.debug('creating code snapshot and adding to sys.path...')
        sys.path.insert(0, check_copy(vc._VALIDATION_CODE_DIR, pjoin(snapshot_dir, 'validation_code')))
        sys.path.insert(0, check_copy(cc._READER_DIR, pjoin(snapshot_dir, 'reader')))
        
        log.debug('starting to run all validations...')
        tasks = TaskDirectory(output_dir)
        run(tasks, validations_to_run, catalogs_to_run, log)
        
        log.debug('creating summary plots...')
        call_summary_plot(tasks, validations_to_run, log)

        log.debug('creating status report...')
        group_by_catalog(tasks, catalogs_to_run, validations_to_run, output_dir)
        write_master_status(master_status, tasks, output_dir)
        report = get_status_report(tasks)
        log.info('All done! Status report:\n' + report)
        log.info('output of this run has been stored in {}'.format(output_dir))
    
    finally:
        os.unlink(pjoin(output_dir, '.lock'))


if __name__ == '__main__':
    main()

