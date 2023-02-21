# DESCQA Validation Tests

This directory hosts the validation tests for DESCQA. This README explains the technical aspects of writting a DESCQA validation test.

## How to write a DESCQA validation test

When you arrive here, you should already know how to access [GCRCatalogs](https://github.com/LSSTDESC/gcr-catalogs) and have a Jupyter notebook or a Python script that runs the test you want to integrate into DESCQA.

Since the DESCQA framework takes care of *running* your test, when you implement a DESCQA, you only implement a class with specific methods that contain the test. In other words, consider a "DESCQA manager" who will run the following code:

```python
with open('configs/your_test_name.yaml') as f:
    test_options = yaml.safe_load(f)

test = YourTestClass(**test_options)

for catalog_name in available_catalogs:
    catalog_instance = GCRCatalogs.load_catalog(catalog_name)
    output_dir = os.path.join(base_output_dir, catalog_name)
    test_result = test.run_on_single_catalog(catalog_instance, catalog_name, output_dir)

test.conclude_test(base_output_dir)
```

Your job is to implement `YourTestClass`, which would be a subclass of `BaseValidationTest`, and to implement the following member methods in `YourTestClass`:

- `__init__`: Should set up the test (getting configs, reading in data etc.).
- `run_on_single_catalog`: Should run the test on one catalog. Should return `TestResult` instance.
- `conclude_test`: Should conclude the test (saving summary plots etc.).

See [example_test.py](example_test.py) for an example.


Each validation test is specified by a YAML config file that sits in the `configs` subdirectory. The YAML config file specifies all the test options and the subclass used to run the test.

The `data` subdirectory hosts small data files that validation tests need, and can be accessed by `self.data_dir`.

