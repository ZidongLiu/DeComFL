from experiment_helper.cli_parser import (
    GeneralSetting,
    DataSetting,
    ModelSetting,
    OptimizerSetting,
    DeviceSetting,
    RGESetting,
    FederatedLearningSetting,
    NormalTrainingLoopSetting,
)


class Settings(
    GeneralSetting,
    DataSetting,
    OptimizerSetting,
    DeviceSetting,
    RGESetting,
    ModelSetting,
    FederatedLearningSetting,
    NormalTrainingLoopSetting,
):
    pass


if __name__ == "__main__":
    args = Settings()
    print(args)
