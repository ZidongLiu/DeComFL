from experiment_helper.cli_parser import (
    GeneralSetting,
    DataSetting,
    LoraSetting,
    ModelSetting,
    OptimizerSetting,
    DeviceSetting,
    RGESetting,
    FederatedLearningSetting,
)


class Settings(
    GeneralSetting,
    DataSetting,
    LoraSetting,
    OptimizerSetting,
    DeviceSetting,
    RGESetting,
    ModelSetting,
    FederatedLearningSetting,
):
    pass


if __name__ == "__main__":
    args = Settings()
    print(args)
