# DESCQA Validation Tests

This directory hosts the validation tests for DESCQA.

The `configs` subdirectory hosts all validation test config YAML files,
and the `data` subdirectory hosts small data files that validation tests need.

To create a new test, inherit `BaseValidationTest` and implement
the following member methods:

- `__init__`: Should set up the test (getting configs, reading in data etc.). Will be called once.
- `run_on_single_catalog`: Should run the test on one catalog. Should return `TestResult` instance. Will be called once for each catalog.
- `conclude_test`: Should conclude the test (saving summary plots etc.). Will be called once when all catalogs have run.

See [example_test.py](example_test.py) for an example.