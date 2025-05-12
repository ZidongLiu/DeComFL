from experiment_helper.cli_parser import (
    FOFLSetting,
    GeneralSetting,
    FederatedLearningSetting,
)


class CliSetting(
    GeneralSetting,
    FederatedLearningSetting,
    FOFLSetting,
):
    """
    This is a replacement for regular argparse module.
    We used a third party library pydantic_setting to make command line interface easier to manage.
    Example:
    if __name__ == "__main__":
        args = CliSetting()

    args will have all parameters defined by all components.
    """

    pass


if __name__ == "__main__":
    args = CliSetting()
    print(args)
