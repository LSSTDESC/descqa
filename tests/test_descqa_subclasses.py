import descqa

def test_subclass_name_in_config():
    for validation_name, validation_config in descqa.available_validations.items():
        assert 'subclass_name' in validation_config, "{}.yaml has no `subclass_name`".format(validation_name)

def test_subclass_importable():
    validations = set(v['subclass_name'] for v in descqa.available_validations.values())
    for validation in validations:
        descqa.register.import_subclass(validation, 'descqa', descqa.BaseValidationTest)
