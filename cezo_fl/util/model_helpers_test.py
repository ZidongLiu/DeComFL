from types import GeneratorType

from torch import nn

from cezo_fl.util import model_helpers


def test_get_trainable_model_parameters():
    sample_model = nn.Sequential(
        nn.Conv2d(1, 20, 5),
        nn.Conv2d(20, 64, 5),
        nn.Conv2d(64, 30, 5),
        nn.Conv2d(20, 64, 5),
    )
    sample_model[0].weight.requires_grad = False
    sample_model[2].weight.requires_grad = False

    # check if it's generator
    assert isinstance(model_helpers.get_trainable_model_parameters(sample_model), GeneratorType)

    # check if it can get all parameters that requires grad
    trainable_parameters_list = list(model_helpers.get_trainable_model_parameters(sample_model))
    # use is to check if 2 references are identical
    assert trainable_parameters_list[0] is sample_model[0].bias
    assert trainable_parameters_list[1] is sample_model[1].weight
    assert trainable_parameters_list[2] is sample_model[1].bias
    assert trainable_parameters_list[3] is sample_model[2].bias
    assert trainable_parameters_list[4] is sample_model[3].weight
    assert trainable_parameters_list[5] is sample_model[3].bias
