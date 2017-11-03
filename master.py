from __future__ import print_function
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
import yaml

pjoin = os.path.join


try:
    basestring
except NameError:
    basestring = str


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


def process_config(config_dir, set_data_dir=None):
    d = dict()

    for f in os.listdir(config_dir):
        if f.startswith('_') or not f.lower().endswith('.yaml'):
            continue

        with open(pjoin(config_dir, f)) as fp:
            d[f[:-5]] = yaml.load(fp.read())

    if set_data_dir:
        for c in d.values():
            c['base_data_dir'] = set_data_dir

    return d


def select_subset(d, keys_wanted=None):
    if keys_wanted is None:
        return d

    keys_wanted = set(keys_wanted)
    keys_available = list(d.keys())
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
        return self._path[key]

    def result_to_text(self, test_result):
        t = 'SKIPPED' if test_result.skipped else ('PASSED' if test_result.passed else 'FAILED')
        return 'VALIDATION_TEST_' + t

    def set_status(self, validation_name, catalog_name, test_result):
        status = test_result.upper() if isinstance(test_result, basestring) else self.result_to_text(test_result)
        self._status[validation_name][catalog_name] = status
        with open(pjoin(self.get_path(validation_name, catalog_name), 'STATUS'), 'w') as f:
            f.write(status + '\n')
            if hasattr(test_result, 'summary'):
                f.write(test_result.summary + '\n')
            if hasattr(test_result, 'score'):
                f.write('{:.3g}\n'.format(test_result.score))

    def get_status(self, validation_name=None, catalog_name=None):
        if catalog_name:
            return self._status[validation_name][catalog_name]
        elif validation_name:
            return self._status[validation_name]
        else:
            return self._status


def make_all_subdirs(tasks, validations_to_run, catalogs_to_run):
    for validation in validations_to_run:
        os.mkdir(tasks.get_path(validation))
        for catalog in catalogs_to_run:
            os.mkdir(tasks.get_path(validation, catalog))


def run(tasks, validations_to_run, catalogs_to_run, log):
    validation_instance_cache = {} # to cache validation instance

    def write_to_traceback(msg):
        with open(pjoin(final_output_dir, 'traceback.log'), 'a') as f:
            f.write(msg)

    # loading catalog usually takes longer, so we put catalogs_to_run in outer loop
    for catalog_name in catalogs_to_run:
        # check if there's still valid validations to run
        if all(isinstance(validation_instance_cache.get(v), basestring) for v in validations_to_run):
            log.debug('skipping "{}" catalog as there are errors in all validations'.format(catalog_name))
            continue

        # try loading the catalog
        log.info('loading "{}" catalog...'.format(catalog_name))
        catcher = ExceptionAndStdStreamCatcher()
        with CatchExceptionAndStdStream(catcher):
            gc = GCRCatalogs.load_catalog(catalog_name)
        if catcher.has_exception:
            log.error('error occurred when loading "{}" catalog...'.format(catalog_name))
            log.debug('stdout/stderr and traceback:\n' + catcher.output)
            gc = catcher.output
        elif catcher.output:
            log.debug('stdout/stderr while loading "{}" catalog:\n'.format(catalog_name) + catcher.output)

        # loop over validations_to_run
        for validation_name, validation in validations_to_run.iteritems():
            # get the final output path, set test name
            final_output_dir = tasks.get_path(validation_name, catalog_name)
            validation['test_name'] = validation_name

            # if gc is an error message, log it and abort
            if isinstance(gc, basestring):
                write_to_traceback(gc)
                tasks.set_status(validation_name, catalog_name, 'LOAD_CATALOG_ERROR')
                continue

            # try loading ValidationTest class/instance
            if validation_name not in validation_instance_cache:
                catcher = ExceptionAndStdStreamCatcher()
                with CatchExceptionAndStdStream(catcher):
                    ValidationTest = quick_import(validation['subclass_name'])
                    vt = ValidationTest(**validation)
                if catcher.has_exception:
                    log.error('error occurred when preparing "{}" test'.format(validation_name))
                    log.debug('stdout/stderr and traceback:\n' + catcher.output)
                    vt = catcher.output
                elif catcher.output:
                    log.debug('stdout/stderr while preparing "{}" test:\n'.format(validation_name) + catcher.output)

                # cache the ValidationTest instance for future use
                validation_instance_cache[validation_name] = vt

            else:
                vt = validation_instance_cache[validation_name]

            # if vt is an error message, log it and abort
            if isinstance(vt, basestring):
                write_to_traceback(vt)
                tasks.set_status(validation_name, catalog_name, 'VALIDATION_TEST_MODULE_ERROR')
                continue

            # run validation test
            catcher = ExceptionAndStdStreamCatcher()
            with CatchExceptionAndStdStream(catcher):
                result = vt.run_validation_test(gc, catalog_name, final_output_dir)

            if catcher.output:
                write_to_traceback(catcher.output)
                if catcher.has_exception:
                    log.error('error occurred when running "{}" test on "{}" catalog...'.format(validation_name, catalog_name))
                    log.debug('stdout/stderr and traceback:\n' + catcher.output)
                    tasks.set_status(validation_name, catalog_name, 'RUN_VALIDATION_TEST_ERROR')
                    continue
                else:
                    log.debug('stdout/stderr while running "{}" test on "{}" catalog:\n'.format(validation_name, catalog_name) + catcher.output)

            tasks.set_status(validation_name, catalog_name, result)
            log.info('{} "{}" test on "{}" catalog'.format('skipping' if result.skipped else 'finishing', validation_name, catalog_name))


    log.debug('creating summary plots...')
    for validation in validations_to_run:
        if validation not in validation_instance_cache:
            continue

        catalog_list = []
        for catalog, status in tasks.get_status(validation).iteritems():
            if status.endswith('PASSED') or status.endswith('FAILED'):
                catalog_list.append((catalog, tasks.get_path(validation, catalog)))

        if not catalog_list:
            continue

        catalog_list.sort(key=lambda x: x[0])

        vt = validation_instance_cache[validation]
        catcher = ExceptionAndStdStreamCatcher()
        with CatchExceptionAndStdStream(catcher):
            vt.plot_summary(pjoin(tasks.get_path(validation), 'summary_plot.png'), catalog_list)

        if catcher.has_exception:
            log.error('error occurred when generating summary plot for "{}" test...'.format(validation))
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


def make_argpath_absolute(args):
    args.root_output_dir = os.path.abspath(os.path.expanduser(args.root_output_dir))
    args.source_dir = os.path.abspath(os.path.expanduser(args.source_dir))
    if args.gcr_catalogs_path_overwrite:
        args.gcr_catalogs_path_overwrite = os.path.abspath(os.path.expanduser(args.gcr_catalogs_path_overwrite))
    args.validation_config_dir = pjoin(args.source_dir, os.path.expanduser(args.validation_config_dir))
    args.validation_code_dir = pjoin(args.source_dir, os.path.expanduser(args.validation_code_dir))
    args.validation_data_dir = pjoin(args.source_dir, os.path.expanduser(args.validation_data_dir))


def main(args):
    log = create_logger(verbose=args.verbose)

    master_status = {}
    master_status['user'] = get_username()
    master_status['start_time'] = time.time()
    if args.comment:
        master_status['comment'] = args.comment

    make_argpath_absolute(args)

    log.debug('Importing GCR Catalogs...')
    if args.gcr_catalogs_path_overwrite:
        sys.path.insert(0, args.gcr_catalogs_path_overwrite)
    global GCRCatalogs
    GCRCatalogs = importlib.import_module('GCRCatalogs')
    if args.gcr_catalogs_path_overwrite:
        del sys.path[0]

    if args.list_catalogs:
        print('-'*50)
        for c in sorted(GCRCatalogs.get_available_catalogs()):
            print(c)
        print('-'*50)
        sys.exit(0)

    log.debug('creating output directory...')
    output_dir = make_output_dir(args.root_output_dir, args.subdir)
    open(pjoin(output_dir, '.lock'), 'w').close()

    try: # we want to remove ".lock" file even if anything went wrong

        snapshot_dir = pjoin(output_dir, '_snapshot')
        os.mkdir(snapshot_dir)
        log.info('output of this run is stored in {}'.format(output_dir))

        log.debug('copying config files...')
        check_copy(args.validation_config_dir, pjoin(snapshot_dir, 'validation_configs'))

        log.debug('processing config files...')
        validations_to_run = select_subset(process_config(args.validation_config_dir, args.validation_data_dir), args.validations_to_run)
        catalogs_to_run = select_subset(GCRCatalogs.get_available_catalogs(), args.catalogs_to_run)
        if not validations_to_run or not catalogs_to_run:
            raise ValueError('not thing to run...')

        log.debug('creating code snapshot and adding to sys.path...')
        sys.path.insert(0, check_copy(args.validation_code_dir, pjoin(snapshot_dir, 'validation_code')))

        log.debug('starting to run all validations...')
        tasks = TaskDirectory(output_dir)
        make_all_subdirs(tasks, validations_to_run, catalogs_to_run)
        run(tasks, validations_to_run, catalogs_to_run, log)

        log.debug('creating status report...')
        group_by_catalog(tasks, catalogs_to_run, validations_to_run, output_dir)
        write_master_status(master_status, tasks, output_dir)
        report = get_status_report(tasks)
        log.info('All done! Status report:\n' + report)
        log.info('Web output: https://portal.nersc.gov/project/lsst/descqa/v2/www/index.cgi?run={}'.format(os.path.basename(output_dir)))

    finally:
        os.unlink(pjoin(output_dir, '.lock'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_output_dir',
            help='Output directory (where the web interface runs on). A sub directory named with the current date will be created within.')
    parser.add_argument('--no-subdir', dest='subdir', action='store_false',
            help='If set, no sub directory will be created. In this case, `root_output_dir` must not yet exist.')

    parser.add_argument('-v', '--verbose', action='store_true',
            help='display all debug messages')
    parser.add_argument('-m', '--comment',
            help='attach a comment to this run')

    parser.add_argument('-l', '--lc', '--list-catalogs', dest='list_catalogs', action='store_true',
            help='Just list available catalogs. Runs nothing!')

    parser.add_argument('-t', '--rv', '--validations-to-run', dest='validations_to_run', metavar='VALIDATION', nargs='+',
            help='run only a subset of validations')
    parser.add_argument('-c', '--rc', '--catalogs-to-run', dest='catalogs_to_run', metavar='CATALOG', nargs='+',
            help='run only a subset of catalogs')

    parser.add_argument('--validation-config-dir', default='validation_configs',
            help='validation config file (default: validation_configs)')
    parser.add_argument('--validation-code-dir', default='validation_code',
            help='validation code directory (default: validation_code)')
    parser.add_argument('--validation-data-dir', default='validation_data',
            help='validation data directory (default: validation_data)')

    parser.add_argument('-p', '--gcr-catalogs-path-overwrite',
            help='a path from which Python can import the module `GCRCatalogs`')

    parser.add_argument('--source-dir', default='.',
            help='source directory (default: current working directory)')
    args = parser.parse_args()
    main(args)
