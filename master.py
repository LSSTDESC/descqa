import os
import sys
import shutil
import time
import logging
import traceback
import StringIO
import importlib
import argparse
import collections


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


def process_config(config_dict, keys_to_keep=None):
    d = {k: config_dict[k] for k in config_dict if not k.startswith('_')}
    if keys_to_keep is None:
        return d

    keys_to_keep = set(keys_to_keep)
    if not all(k in d for k in keys_to_keep):
        raise ValueError("Not all required entries ({}) are presented in ({})...".format(\
                ', '.join(keys_to_keep), ', '.join(d.keys())))

    return {k: d[k] for k in d if k in keys_to_keep}


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


def make_all_subdirs(validations_to_run, catalogs_to_run, output_dir):
    for validation in validations_to_run:
        validation_dir = pjoin(output_dir, validation)
        os.mkdir(validation_dir)
        for catalog in catalogs_to_run:
            catalog_output_dir = pjoin(validation_dir, catalog)
            os.mkdir(catalog_output_dir)


def run(validations_to_run, catalogs_to_run, output_dir, log):
    validation_instance_cache = {} # to cache valiation instance
    test_kwargs_dict = collections.defaultdict(dict) # to store all used kwargs
    status_dict = collections.defaultdict(dict) # to store run status

    def set_status(status):
        validation_name, catalog_name = final_output_dir.rstrip(os.path.sep).rsplit(os.path.sep)[-2:]
        status = status.strip().upper()
        if not any(status.endswith(t) for t in ('PASSED', 'FAILED', 'ERROR', 'SKIPPED')):
            raise ValueError('status message not set correctly!')
        with open(pjoin(final_output_dir, 'STATUS'), 'w') as f:
            f.write(status + '\n')
        status_dict[validation_name][catalog_name] = status

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
            gc = Reader(catalog.file)
        if catcher.has_exception:
            log.error('error occured when loading "{}" catalog...'.format(catalog_name))
            log.debug('stdout/stderr and traceback:\n' + catcher.output)
            gc = catcher.output
        elif catcher.output:
            log.debug('stdout/stderr while loading "{}" catalog:\n'.format(catalog_name) + catcher.output)

        # loop over validations_to_run
        for validation_name, validation in validations_to_run.iteritems():
            # get the final output path, set test kwargs
            final_output_dir = pjoin(output_dir, validation_name, catalog_name)
            test_kwargs = validation.kwargs.copy()
            test_kwargs['test_name'] = validation_name

            # if gc is an error message, log it and abort
            if isinstance(gc, basestring):
                write_to_traceback(gc)
                set_status('LOAD_CATALOG_ERROR')
                continue

            # try loading ValidationTest class/instance
            if validation_name not in validation_instance_cache:
                catcher = ExceptionAndStdStreamCatcher()
                with CatchExceptionAndStdStream(catcher):
                    ValidationTest = quick_import(validation.module)
                    vt = ValidationTest(**test_kwargs)
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
                set_status('VALIDATION_TEST_MODULE_ERROR')
                continue

            # run validation test
            catcher = ExceptionAndStdStreamCatcher()
            with CatchExceptionAndStdStream(catcher):
                error_code = vt.run_validation_test(gc, catalog_name, final_output_dir)

            if catcher.output:
                write_to_traceback(catcher.output)
                if catcher.has_exception:
                    log.error('error occured when running "{}" test on "{}" catalog...'.format(validation_name, catalog_name))
                    log.debug('stdout/stderr and traceback:\n' + catcher.output)
                    set_status('RUN_VALIDATION_TEST_ERROR')
                else:
                    log.debug('stdout/stderr while running "{}" test on "{}" catalog:\n'.format(validation_name, catalog_name) + catcher.output)
                
            if not catcher.has_exception:
                if error_code == 1:
                    set_status('VALIDATION_TEST_FAILED')
                elif error_code:
                    set_status('VALIDATION_TEST_SKIPPED')
                else:
                    set_status('VALIDATION_TEST_PASSED')

                if error_code == 0 or error_code == 1:
                    test_kwargs['catalog_name'] = catalog_name
                    test_kwargs['base_output_dir'] = final_output_dir
                    test_kwargs_dict[validation_name][catalog_name] = test_kwargs.copy()
                    log.info('finishing "{}" test on "{}" catalog'.format(validation_name, catalog_name))
                else:
                    log.info('skipping "{}" test on "{}" catalog'.format(validation_name, catalog_name))

    # now back outside the two loops, return status
    return status_dict, test_kwargs_dict


def call_summary_plot(test_kwargs_dict, validations_to_run, output_dir, log):
    for validation in test_kwargs_dict:
        module_name = validations_to_run[validation].module
        
        catcher = ExceptionAndStdStreamCatcher()
        with CatchExceptionAndStdStream(catcher):
            plot_summary_func = getattr(importlib.import_module(module_name), 'plot_summary')
            plot_summary_func(pjoin(output_dir, validation, 'summary_plot.png'), test_kwargs_dict[validation].values())

        if catcher.has_exception:
            log.error('error occured when generating summary plot for "{}" test...'.format(validation))
            log.debug('stdout/stderr and traceback:\n' + catcher.output)
        elif catcher.output:
            log.debug('stdout/stderr while generating summary plot for "{}" test:\n'.format(validation) + catcher.output)

        if catcher.output:
            with open(pjoin(output_dir, validation, 'summary_plot.log'), 'w') as f:
                f.write(catcher.output)

            
def get_status_report(status):

    report = StringIO.StringIO()
    
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


def interfacing_webview(status, output_dir):
    with open(pjoin(output_dir, 'errors'), 'w') as f_top:
        for validation in status:
            total = len(status[validation])
            counter = collections.Counter(status[validation].values())
            s = '; '.join(('{}/{} {}'.format(v, total, k) for k, v in counter.iteritems()))
            f_top.write('{} - {}\n'.format(validation, s))
            with open(pjoin(output_dir, validation, 'errors'), 'w') as f_this:
                f_this.write('0\n0\n{}\n{}\n{}\n'.format(\
                        sum(counter[k] for k in counter if k.endswith('ERROR')), 
                        sum(counter[k] for k in counter if k.endswith('FAILED')), 
                        total))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_output_dir')
    parser.add_argument('--no-subdir', dest='subdir', action='store_false')
    parser.add_argument('--validation-config', default='config_validation.py')
    parser.add_argument('--catalog-config', default='config_catalog.py')
    parser.add_argument('--validations-to-run', metavar='VALIDATION', nargs='+')
    parser.add_argument('--catalogs-to-run', metavar='CATALOG', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    log = create_logger(verbose=args.verbose)

    log.debug('creating output directory...')
    output_dir = make_output_dir(args.root_output_dir, args.subdir)
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

    log.debug('making all sub directories...')
    make_all_subdirs(validations_to_run, catalogs_to_run, output_dir)
    
    log.debug('starting to run all validations...')
    status, test_kwargs = run(validations_to_run, catalogs_to_run, output_dir, log)
    
    log.debug('creating summary plots...')
    call_summary_plot(test_kwargs, validations_to_run, output_dir, log)

    log.debug('creating status report...')
    interfacing_webview(status, output_dir)
    report = get_status_report(status)
    log.info('All done! Status report:\n' + report)

if __name__ == '__main__':
    main()

