from experiment_helper.cli_parser import GeneralSetting

import sys


def test_general_setting():
    # default
    sys.argv = ["simplified_test.py"]
    setting = GeneralSetting()
    assert setting.seed == 365
    assert setting.log_to_tensorboard is None

    # some change
    sys.argv = ["simplified_test.py", "--seed=123", "--log-to-tensorboard=./asd/dsa"]
    setting = GeneralSetting()
    assert setting.seed == 123
    assert setting.log_to_tensorboard == "./asd/dsa"
