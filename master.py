import os
import sys
import shutil
import time
import logging
import traceback
import importlib
import argparse


output_filenames = {
        'catalog-level':{
                'figure': 'figure.png',
                'catalog': 'plot_output.txt',
                'validation': 'validation_output.txt',
                'log': 'bcc_output',
                'summary': 'test_output',
                'status': 'errors',
                'exec_detail': 'run_summary',
                },
        'validation-level':{
                'status': 'errors',
                'log': 'traceback.log',
                },
        'run-level':{
                'status': 'errors',
                'log': 'flash_test.log',
                },
}


pjoin = os.path.join


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
        raise ValueError("required ")#TODO

    return [d for d in config if d[index_key] in subset_to_keep]


def create_logger():
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
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


def run(validations_to_run, catalogs_to_run, output_dir, validation_data_dir, catalog_data_dir, log):
    for validation in validations_to_run:
        log.info('preparing "{}" validation...'.format(validation['name']))
        try:
            validation_dir = pjoin(output_dir, validation['name'])
            os.mkdir(validation_dir)
            ValidationTest = quick_import(validation['module'])
            data_dir = pjoin(validation_data_dir, validation['data_dir'])
            vt = ValidationTest(validation.get('test_args', {}), data_dir, validation.get('data_args', {}))

            status_count = {'test_fail': 0, 'exec_fail': 0}
            for catalog in catalogs_to_run:
                try:
                    log.info('running on "{}" catalog...'.format(catalog['name']))
                    start_time = time.time()
                    
                    catalog_output_dir = pjoin(validation_dir, catalog['name'])
                    os.mkdir(catalog_output_dir)
                    output_paths = {k: pjoin(catalog_output_dir, v) for k, v in output_filenames['catalog-level'].iteritems()}
                    
                    Reader = quick_import(catalog['reader'])
                    galaxy_catalog = Reader(pjoin(catalog_data_dir, catalog['file']))
                    success = vt.run_validation_test(galaxy_catalog, catalog['name'], output_paths)
                    
                    if not success:
                        status_count['test_fail'] += 1

                    # writing out files that the webserver needs

                    with open(output_paths['exec_detail'], 'w') as f:
                        f.write('numProcs: 1\nwallClockTime: {0} second\nnumOutputFiles: {1}\nnumTheoryFiles: {1}\n'.format(\
                                time.time()-start_time, int(success)))

                    with open(output_paths['status'], 'w') as f:
                        f.write('0\n0\n')

                except:
                    # catalog-level error
                    log.error('something wrong happened when running "{}" on "{}"'.format(validation['name'], catalog['name']))
                    with open(pjoin(catalog_output_dir, 'traceback.log'), 'w') as f:
                        traceback.print_exc(file=f)
                    status_count['exec_fail'] += 1

            with open(pjoin(validation_dir, output_filenames['validation-level']['status']), 'w') as f:
                f.write('0\n0\n0\n{}\n{}\n'.format(len(catalogs_to_run)-sum(status_count.values()), len(catalogs_to_run)))
        
            with open(pjoin(output_dir, output_filenames['run-level']['status']), 'a') as f:
                f.write('{0} - {1[exec_fail]}/{2} failed in execution; {1[test_fail]}/{2} failed in testing\n'.format(\
                        validation['name'], status_count, len(catalogs_to_run)))

        except:
            # test-level error
            log.error('something wrong happened when preparing "{}" validation'.format(validation['name']))
            with open(pjoin(validation_dir, 'traceback.log'), 'w') as f:
                traceback.print_exc(file=f)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_output_dir')
    parser.add_argument('--no-subdir', dest='subdir', action='store_false')
    parser.add_argument('--validation-config', default='config_validation.py')
    parser.add_argument('--catalog-config', default='config_catalog.py')
    parser.add_argument('--validations-to-run', metavar='VALIDATION', nargs='+')
    parser.add_argument('--catalogs-to-run', metavar='CATALOG', nargs='+')
    args = parser.parse_args()

    log = create_logger()

    log.info('creating output directory...')
    output_dir = make_output_dir(args.root_output_dir, args.subdir)
    snapshot_dir = pjoin(output_dir, 'snapshot')
    os.mkdir(snapshot_dir)
    log.info('output of this run stored in {}'.format(output_dir))

    log.info('copying config files...')
    check_copy(args.validation_config, pjoin(snapshot_dir, 'validation_config.py'))
    check_copy(args.catalog_config, pjoin(snapshot_dir, 'catalog_config.py'))

    log.info('importing config files...')
    sys.path.insert(0, snapshot_dir)
    from validation_config import VALIDATION_CONFIG, VALIDATION_CODE_DIR, VALIDATION_DATA_DIR
    from catalog_config import CATALOG_CONFIG, READER_DIR, CATALOG_DIR
    del sys.path[0]

    log.info('processing config files...')
    validations_to_run = process_config(VALIDATION_CONFIG, ['module'], args.validations_to_run)
    catalogs_to_run = process_config(CATALOG_CONFIG, ['file', 'reader'], args.catalogs_to_run)

    log.info('creating code snapshot...')
    VALIDATION_CODE_DIR = check_copy(VALIDATION_CODE_DIR, pjoin(snapshot_dir, 'validation_code'))
    READER_DIR = check_copy(READER_DIR, pjoin(snapshot_dir, 'reader'))
    
    log.info('adding module paths to sys.path...')
    sys.path.insert(0, VALIDATION_CODE_DIR)
    sys.path.insert(0, READER_DIR)

    log.info('starting to run validations...')
    run(validations_to_run, catalogs_to_run, output_dir, VALIDATION_DATA_DIR, CATALOG_DIR, log)


if __name__ == '__main__':
    main()

