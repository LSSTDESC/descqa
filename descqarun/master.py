from __future__ import print_function, unicode_literals, absolute_import
import os
import sys
import shutil
import time
import json
import logging
import traceback
import importlib
import argparse
import collections
import fnmatch
import subprocess

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import yaml
from . import config

__all__ = ['main']


_horizontal_rule = '-'*50

pjoin = os.path.join


def make_path_absolute(path):
    return os.path.abspath(os.path.expanduser(path))


def _is_string_like(obj):
    """
    Check whether obj behaves like a string.
    """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True


class CatchExceptionAndStdStream():
    def __init__(self, filenames=None, logger=None, during=None):
        self._logger = logger
        self._filenames = [filenames] if _is_string_like(filenames) else filenames
        self._during = ' when {}'.format(during) if during else ''
        self._stream = StringIO()
        self._stdout = self._stderr = None

    def __enter__(self):
        self._stdout = sys.stdout
        self._stdout.flush()
        sys.stdout = self._stream
        self._stderr = sys.stderr
        self._stderr.flush()
        sys.stderr = self._stream

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._stream.flush()
        has_exception = False
        if exc_type:
            traceback.print_exception(exc_type, exc_value, exc_tb)
            has_exception = True
        output = self._stream.getvalue().strip()
        self._stream.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr

        if self._logger:
            if has_exception:
                self._logger.error('Exception occurred{}. Below are stdout/stderr and traceback:\n{}'.format(self._during, output))
            elif output:
                self._logger.debug('Below are stdout/stderr{}:\n{}'.format(self._during, output))

        if self._filenames and output:
            for filename in self._filenames:
                with open(filename, 'a') as f:
                    f.write(output)
                    f.write('\n')

        return True


def create_logger(verbose=False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logFormatter = logging.Formatter('[%(levelname)-5.5s][%(asctime)s] %(message)s')
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    return logger


def record_version(module, version, record_dict=None, logger=None):
    if record_dict is None:
        record_dict = dict()
    record_dict[module] = version
    if logger:
        logger.info('Using {} {}'.format(module, version))
    return record_dict


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


def make_output_dir(root_output_dir):
    root_output_dir = make_path_absolute(root_output_dir)
    if not os.path.isdir(root_output_dir):
        raise OSError('{} does not exist'.format(root_output_dir))

    new_dir_name = time.strftime('%Y-%m-%d')
    parent_dir = pjoin(root_output_dir, new_dir_name.rpartition('-')[0])
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
        subprocess.check_call(['chmod', 'a+rx,o-w', parent_dir])

    output_dir = pjoin(parent_dir, new_dir_name)
    if os.path.exists(output_dir):
        i = max((int(s.partition('_')[-1] or 0) for s in os.listdir(parent_dir) if s.startswith(new_dir_name)))
        output_dir += '_{}'.format(i+1)

    os.mkdir(output_dir)
    return output_dir


def get_username():
    for k in ('LOGNAME', 'USER', 'LNAME', 'USERNAME'):
        user = os.getenv(k)
        if user:
            return user
    return 'UNKNOWN'


def print_available_and_exit(catalogs, validations):
    print(_horizontal_rule)
    print('Available catalogs')
    print(_horizontal_rule)
    for c in sorted(catalogs):
        print(c, '*' if catalogs[c].get('included_by_default') else '')
    print()

    print(_horizontal_rule)
    print('Available validations')
    print(_horizontal_rule)
    for v in sorted(validations):
        print(v, '*' if validations[v].get('included_by_default') else '')
    print()

    sys.exit(0)


class DescqaTask(object):

    logfile_basename = 'traceback.log'
    config_basename = 'config.yaml'
    status_basename = 'STATUS'

    def __init__(self, output_dir, validations_to_run, catalogs_to_run, logger):
        self.output_dir = output_dir
        self.logger = logger
        self.validations_to_run = self.select_subset(descqa.available_validations, validations_to_run)
        self.catalogs_to_run = self.select_subset(GCRCatalogs.available_catalogs, catalogs_to_run)

        if not self.validations_to_run or not self.catalogs_to_run:
            raise RuntimeError('Nothing to run... Aborted!')

        self._validation_instance_cache = dict()
        self._results = dict()


    @staticmethod
    def select_subset(available, wanted=None):
        if wanted is None:
            available_default = None
            if isinstance(available, dict):
                available_default = [k for k, v in available.items() if v.get('included_by_default')]
            return set(available_default) if available_default else set(available)

        wanted = set(wanted)
        output = set()

        for item in wanted:
            matched = fnmatch.filter(available, item)
            if not matched:
                raise KeyError("{} does not match any available names: {}".format(item, ', '.join(sorted(available))))
            output.update(matched)

        return tuple(sorted(output))


    def get_path(self, validation, catalog=None):
        return pjoin(self.output_dir, validation, catalog) if catalog else pjoin(self.output_dir, validation)


    def make_all_subdirs(self):
        for validation in self.validations_to_run:
            os.mkdir(self.get_path(validation))

            for catalog in self.catalogs_to_run:
                os.mkdir(self.get_path(validation, catalog))

            with open(pjoin(self.get_path(validation), self.config_basename), 'w') as f:
                f.write(yaml.dump(descqa.available_validations[validation], default_flow_style=False))
                f.write('\n')


    def get_description(self, description_key='description'):
        dv = {v: descqa.available_validations[v].get(description_key) for v in self.validations_to_run}
        dc = {c: GCRCatalogs.get_catalog_config(c).get(description_key) for c in self.catalogs_to_run}
        return {'validation_{}'.format(description_key): dv, 'catalog_{}'.format(description_key): dc}


    def get_validation_instance(self, validation):
        if validation not in self._validation_instance_cache:
            logfile = pjoin(self.get_path(validation), self.logfile_basename)
            instance = None
            with CatchExceptionAndStdStream(logfile, self.logger, 'loading validation `{}`'.format(validation)):
                instance = descqa.load_validation(validation)
            if instance is None:
                self.set_result('VALIDATION_TEST_MODULE_ERROR', validation=validation)
            self._validation_instance_cache[validation] = instance
        return self._validation_instance_cache[validation]


    def get_catalog_instance(self, catalog):
        logfile = [pjoin(self.get_path(validation, catalog), self.logfile_basename) for validation in self.validations_to_run]
        instance = None
        with CatchExceptionAndStdStream(logfile, self.logger, 'loading catalog `{}`'.format(catalog)):
            instance = GCRCatalogs.load_catalog(catalog)
        if instance is None:
            self.set_result('LOAD_CATALOG_ERROR', catalog=catalog)
        return instance


    def set_result(self, test_result, validation=None, catalog=None):
        if validation and catalog:
            key = (validation, catalog)
        elif validation:
            for c in self.catalogs_to_run:
                self.set_result(test_result, validation, c)
            return
        elif catalog:
            for v in self.validations_to_run:
                self.set_result(test_result, v, catalog)
            return
        else:
            raise ValueError('Must specify *validation* and/or *catalog*')

        if key in self._results:
            self.logger.debug('Warning: result of {} has been set already!'.format(key))
            return

        if _is_string_like(test_result):
            status = test_result
            test_result = None
        elif hasattr(test_result, 'status_code'):
            status = test_result.status_code
        else:
            status = 'VALIDATION_TEST_{}'.format('SKIPPED' if test_result.skipped else ('PASSED' if test_result.passed else 'FAILED'))

        self._results[key] = (status, test_result)

        with open(pjoin(self.get_path(*key), self.status_basename), 'w') as f:
            if hasattr(test_result, 'status_full'):
                f.write(test_result.status_full + '\n')
            else:
                f.write(status + '\n')
                if getattr(test_result, 'summary', None):
                    f.write(test_result.summary + '\n')
                if getattr(test_result, 'score', None):
                    f.write('{:.3g}'.format(test_result.score) + '\n')


    def get_status(self, validation=None, catalog=None, return_test_result=False):
        if validation and catalog:
            return self._results.get((validation, catalog), (None, None))[int(return_test_result)]
        elif validation:
            return {c: self.get_status(validation, c, return_test_result) for c in self.catalogs_to_run}
        elif catalog:
            return {v: self.get_status(v, catalog, return_test_result) for v in self.validations_to_run}
        else:
            return {v: self.get_status(v, None, return_test_result) for v in self.validations_to_run}


    def check_status(self):
        msg = 'hmmm, something is wrong with the test results!'
        if not all((v, c) in self._results for v in self.validations_to_run for c in self.catalogs_to_run):
            self.logger.error(msg)


    def count_status(self):
        count_by_validation = {v: collections.Counter(self.get_status(validation=v).values()) for v in self.validations_to_run}
        count_by_catalog = {c: collections.Counter(self.get_status(catalog=c).values()) for c in self.catalogs_to_run}
        return count_by_validation, count_by_catalog


    def get_status_report(self):
        report = StringIO()
        for validation in self.validations_to_run:
            report.write(_horizontal_rule + '\n')
            report.write(validation + '\n')
            report.write(_horizontal_rule + '\n')
            l = max(len(catalog) for catalog in self.catalogs_to_run)
            l += 3

            for catalog in self.catalogs_to_run:
                s = self.get_status(validation, catalog)
                report.write('{{:{}}}{{}}\n'.format(l).format(catalog, s))
        report.write(_horizontal_rule + '\n')
        report_content = report.getvalue()
        report.close()
        return report_content


    def run_tests(self):
        run_at_least_one_catalog = False
        for catalog in self.catalogs_to_run:
            catalog_instance = self.get_catalog_instance(catalog)
            if catalog_instance is None:
                continue

            run_at_least_one_catalog = True
            for validation in self.validations_to_run:
                validation_instance = self.get_validation_instance(validation)
                if validation_instance is None:
                    continue

                output_dir_this = self.get_path(validation, catalog)
                logfile = pjoin(output_dir_this, self.logfile_basename)
                msg = 'running validation `{}` on catalog `{}`'.format(validation, catalog)
                self.logger.debug(msg)

                test_result = None
                with CatchExceptionAndStdStream(logfile, self.logger, msg):
                    test_result = validation_instance.run_on_single_catalog(catalog_instance, catalog, output_dir_this)

                self.set_result(test_result or 'RUN_VALIDATION_TEST_ERROR', validation, catalog)

        if not run_at_least_one_catalog:
            msg = 'No valid catalog to run! Abort!'
            self.logger.error(msg)
            raise RuntimeError(msg)


    def conclude_tests(self):
        for validation in self.validations_to_run:
            validation_instance = self.get_validation_instance(validation)
            if validation_instance is None:
                continue

            output_dir_this = self.get_path(validation)
            logfile = pjoin(output_dir_this, self.logfile_basename)
            msg = 'concluding validation test `{}`'.format(validation)
            self.logger.debug(msg)

            with CatchExceptionAndStdStream(logfile, self.logger, msg):
                validation_instance.conclude_test(output_dir_this)


    def run(self):
        self.logger.debug('creating subdirectories in output_dir...')
        self.make_all_subdirs()

        if all(self.get_validation_instance(validation) is None for validation in self.validations_to_run):
            self.logger.info('No valid validation tests. End program.')
            self.check_status()
            return

        self.logger.debug('starting to run all validation tests...')
        self.run_tests()
        self.check_status()

        self.logger.debug('starting to conclude all validation tests...')
        self.conclude_tests()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_output_dir',
            help='Output directory (where the web interface runs on). A sub directory named with the current date will be created within.')

    parser.add_argument('-v', '--verbose', action='store_true',
            help='Display all debug messages')
    parser.add_argument('-m', '--comment',
            help='Attach a comment to this run')

    parser.add_argument('-l', '--list', action='store_true',
            help='Just list available catalogs and validations. Runs nothing!')

    parser.add_argument('-t', '--validations-to-run', dest='validations_to_run', metavar='VALIDATION', nargs='+',
            help='Run only a subset of validations')
    parser.add_argument('-c', '--catalogs-to-run', dest='catalogs_to_run', metavar='CATALOG', nargs='+',
            help='run only a subset of catalogs')

    parser.add_argument('-p', '--insert-sys-path', dest='paths', metavar='PATH', nargs='+',
            help='Insert path(s) to sys.path')

    parser.add_argument('-w', '--web-base-url', metavar='URL', default=config.base_url,
            help='Web interface base URL')

    args = parser.parse_args()

    logger = create_logger(verbose=args.verbose)

    master_status = dict()
    master_status['user'] = get_username()
    master_status['start_time'] = time.time()
    if args.comment:
        master_status['comment'] = args.comment
    master_status['versions'] = dict()

    logger.debug('Importing DESCQA and GCR Catalogs...')
    if args.paths:
        sys.path = [make_path_absolute(path) for path in args.paths] + sys.path

    global GCRCatalogs #pylint: disable=W0601
    GCRCatalogs = importlib.import_module('GCRCatalogs')

    global descqa #pylint: disable=W0601
    descqa = importlib.import_module('descqa')

    record_version('DESCQA', descqa.__version__, master_status['versions'], logger=logger)
    record_version('GCRCatalogs', GCRCatalogs.__version__, master_status['versions'], logger=logger)
    if hasattr(GCRCatalogs, 'GCR'):
        record_version('GCR', GCRCatalogs.GCR.__version__, master_status['versions'], logger=logger)

    if args.list:
        print_available_and_exit(GCRCatalogs.available_catalogs, descqa.available_validations)

    logger.debug('creating root output directory...')
    output_dir = make_output_dir(args.root_output_dir)
    open(pjoin(output_dir, '.lock'), 'w').close()

    try: # we want to remove ".lock" file even if anything went wrong

        logger.info('output of this run is stored in %s', output_dir)
        logger.debug('creating code snapshot...')
        snapshot_dir = pjoin(output_dir, '_snapshot')
        os.mkdir(snapshot_dir)
        check_copy(descqa.__path__[0], pjoin(snapshot_dir, 'descqa'))
        check_copy(GCRCatalogs.__path__[0], pjoin(snapshot_dir, 'GCRCatalogs'))
        if hasattr(GCRCatalogs, 'GCR'):
            check_copy(GCRCatalogs.GCR.__file__, pjoin(snapshot_dir, 'GCR.py'))

        logger.debug('preparing to run validation tests...')
        descqa_task = DescqaTask(output_dir, args.validations_to_run, args.catalogs_to_run, logger)
        master_status.update(descqa_task.get_description())

        logger.info('running validation tests...')
        descqa_task.run()

        logger.debug('finishing up...')
        master_status['status_count'], master_status['status_count_group_by_catalog'] = descqa_task.count_status()
        master_status['end_time'] = time.time()
        with open(pjoin(output_dir, 'STATUS.json'), 'w') as f:
            json.dump(master_status, f, indent=True)

        logger.info('All done! Status report:\n%s', descqa_task.get_status_report())

    finally:
        os.unlink(pjoin(output_dir, '.lock'))
        subprocess.check_call(['chmod', '-R', 'a+rX,o-w', output_dir])
        logger.info('Web output: %s?run=%s', args.web_base_url, os.path.basename(output_dir))


if __name__ == '__main__':
    main()
