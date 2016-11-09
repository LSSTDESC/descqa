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


output_filenames = {
     'figure': 'figure.png',
     'catalog': 'catalog_output.txt',
     'validation': 'validation_output.txt',
     'log': 'runtime.log',
     'summary': 'summary.txt',
}


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


def process_config(config, additional_required_keys=[], subset_to_keep=None, index_key='name'):
    required_keys = set(additional_required_keys)
    required_keys.add(index_key)
    indices = set()
    for d in config:
        if not all(k in d for k in required_keys):
            raise ValueError('Each entry must have {} set'.format(', '.join(required_keys)))
        if d[index_key] in indices:
            raise ValueError('Each entry must have a unique value for {}'.format(index_key))
        indices.add(d[index_key])
    
    if subset_to_keep is None:
        return config

    subset_to_keep = set(subset_to_keep)
    if not all(i in indices for i in subset_to_keep):
        raise ValueError("Not all required entries ({}) are presented in ({})...".format(\
                ', '.join(subset_to_keep), ', '.join(indices)))

    return [d for d in config if d[index_key] in subset_to_keep]


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
        validation_dir = pjoin(output_dir, validation['name'])
        os.mkdir(validation_dir)
        for catalog in catalogs_to_run:
            catalog_output_dir = pjoin(validation_dir, catalog['name'])
            os.mkdir(catalog_output_dir)


def run(validations_to_run, catalogs_to_run, output_dir, validation_data_dir, catalog_data_dir, log):
    validation_instance_cache = {}
    
    status_dict = collections.defaultdict(dict)
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
    for catalog in catalogs_to_run:
        # try loading the catalog
        log.info('loading "{}" catalog...'.format(catalog['name']))
        catcher = ExceptionAndStdStreamCatcher()
        with CatchExceptionAndStdStream(catcher):
            Reader = quick_import(catalog['reader'])
            gc = Reader(pjoin(catalog_data_dir, catalog['file']))
        if catcher.has_exception:
            log.error('error occured when loading "{}" catalog...'.format(catalog['name']))
            log.debug('stdout/stderr and traceback:\n' + catcher.output)
            gc = catcher.output
        elif catcher.output:
            log.debug('stdout/stderr while loading "{}" catalog:\n'.format(catalog['name']) + catcher.output)

        # loop over validations_to_run
        for validation in validations_to_run:
            # get the final output path, set traceback file path
            final_output_dir = pjoin(output_dir, validation['name'], catalog['name'])

            # if gc is an error message, log it and abort
            if isinstance(gc, basestring):
                write_to_traceback(gc)
                set_status('LOAD_CATALOG_ERROR')
                continue

            # try loading ValidationTest class/instance
            if validation['name'] not in validation_instance_cache:
                catcher = ExceptionAndStdStreamCatcher()
                with CatchExceptionAndStdStream(catcher):
                    ValidationTest = quick_import(validation['module'])
                    vt = ValidationTest(validation.get('test_args', {}), 
                                        pjoin(validation_data_dir, validation['data_dir']),
                                        validation.get('data_args', {}))
                if catcher.has_exception:
                    log.error('error occured when preparing "{}" test'.format(validation['name']))
                    log.debug('stdout/stderr and traceback:\n' + catcher.output)
                    vt = catcher.output
                elif catcher.output:
                    log.debug('stdout/stderr while preparing "{}" test:\n'.format(validation['name']) + catcher.output)

                # cache the ValidationTest instance for future use
                validation_instance_cache[validation['name']] = vt
            
            else:
                vt = validation_instance_cache[validation['name']]
            
            #if vt is an error message, log it and abort
            if isinstance(vt, basestring): 
                write_to_traceback(vt)
                set_status('VALIDATION_TEST_MODULE_ERROR')
                continue

            # set output paths for run_validation_test
            output_paths = {k: pjoin(final_output_dir, v) for k, v in output_filenames.iteritems()}
            
            # run validation test
            catcher = ExceptionAndStdStreamCatcher()
            with CatchExceptionAndStdStream(catcher):
                error_code = vt.run_validation_test(gc, catalog['name'], output_paths)

            if catcher.output:
                write_to_traceback(catcher.output)
                if catcher.has_exception:
                    log.error('error occured when running "{}" test on "{}" catalog...'.format(validation['name'], catalog['name']))
                    log.debug('stdout/stderr and traceback:\n' + catcher.output)
                    set_status('RUN_VALIDATION_TEST_ERROR')
                else:
                    log.debug('stdout/stderr while running "{}" test on "{}" catalog:\n'.format(validation['name'], catalog['name']) + catcher.output)
                
            if not catcher.has_exception:
                if error_code == 1:
                    set_status('VALIDATION_TEST_FAILED')
                elif error_code:
                    set_status('VALIDATION_TEST_SKIPPED')
                else:
                    set_status('VALIDATION_TEST_PASSED')

                log.info('finishing "{}" test on "{}" catalog'.format(validation['name'], catalog['name']))

    # now back outside the two loops, return status
    return status_dict


def get_status_report(status, validations_to_run, catalogs_to_run):

    l = max(len(catalog['name']) for catalog in catalogs_to_run)
    l += 3

    report = StringIO.StringIO()
    
    for validation in validations_to_run:
        report.write('-'*50 + '\n')
        report.write(validation['name'] + '\n')
        report.write('-'*50 + '\n')
        
        for catalog in catalogs_to_run:
            s = status[validation['name']][catalog['name']]
            report.write('{{:{}}}{{}}\n'.format(l).format(catalog['name'], s))

    report.write('-'*50 + '\n')
    
    report_content = report.getvalue()
    report.close()

    return report_content


def interfacing_webview(status, validations_to_run, output_dir):
    with open(pjoin(output_dir, 'errors'), 'w') as f_top:
        for validation in validations_to_run:
            total = len(status[validation['name']])
            counter = collections.Counter(status[validation['name']].values())
            s = '; '.join(('{}/{} {}'.format(v, total, k) for k, v in counter.iteritems()))
            f_top.write('{} - {}\n'.format(validation['name'], s))
            with open(pjoin(output_dir, validation['name'], 'errors'), 'w') as f_this:
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
    from validation_config import VALIDATION_CONFIG, VALIDATION_CODE_DIR, VALIDATION_DATA_DIR
    from catalog_config import CATALOG_CONFIG, READER_DIR, CATALOG_DIR
    del sys.path[0]

    log.debug('processing config files...')
    validations_to_run = process_config(VALIDATION_CONFIG, ['module'], args.validations_to_run)
    catalogs_to_run = process_config(CATALOG_CONFIG, ['file', 'reader'], args.catalogs_to_run)

    log.debug('creating code snapshot...')
    VALIDATION_CODE_DIR = check_copy(VALIDATION_CODE_DIR, pjoin(snapshot_dir, 'validation_code'))
    READER_DIR = check_copy(READER_DIR, pjoin(snapshot_dir, 'reader'))
    
    log.debug('adding module paths to sys.path...')
    sys.path.insert(0, VALIDATION_CODE_DIR)
    sys.path.insert(0, READER_DIR)

    log.debug('making all sub directories...')
    make_all_subdirs(validations_to_run, catalogs_to_run, output_dir)
    
    log.debug('starting to run all validations...')
    status = run(validations_to_run, catalogs_to_run, output_dir, VALIDATION_DATA_DIR, CATALOG_DIR, log)
    
    log.debug('creating status report...')
    interfacing_webview(status, validations_to_run, output_dir)
    report = get_status_report(status, validations_to_run, catalogs_to_run)
    log.info('All done! Status report:\n' + report)

if __name__ == '__main__':
    main()

