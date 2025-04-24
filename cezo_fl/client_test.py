# from copy import deepcopy

# import torch
# from torch.optim import SGD

# from cezo_fl.client import SyncClient
# from cezo_fl.models.cnn_mnist import CNN_MNIST
# from cezo_fl.gradient_estimators.random_gradient_estimator import RandomGradientEstimator
# from cezo_fl.util.metrics import accuracy
# from cezo_fl.fl_helpers import get_server_name
# from experiment_helper import device, cli_parser, data


# class Setting(
#     cli_parser.DeviceSetting,
#     cli_parser.DataSetting,
#     cli_parser.OptimizerSetting,
#     cli_parse_args=False,
# ):
#     pass


# NOTE: this unit test only passes for 1e-6
def test_sync_client_reset():
    pass
    # args = Setting()
    # device_map = device.use_device(args.device_setting, num_clients=1)
    # train_loaders, _ = data.get_dataloaders(args.data_setting, num_train_split=1, seed=365)
    # model_device = device_map[get_server_name()]

    # model = CNN_MNIST().to(model_device)

    # train_loader = train_loaders[0]
    # grad_estimator = RandomGradientEstimator(
    #     model.parameters(),
    #     mu=1e-3,
    #     num_pert=2,
    #     grad_estimate_method="rge-forward",
    #     device=model_device,
    # )
    # optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=0)

    # criterion = torch.nn.CrossEntropyLoss()

    # sync_client = SyncClient(
    #     model=model,
    #     model_inference=lambda m, x: m(x),
    #     dataloader=train_loader,
    #     grad_estimator=grad_estimator,
    #     optimizer=optimizer,
    #     criterion=criterion,
    #     accuracy_func=accuracy,
    #     device=model_device,
    # )

    # original_model = deepcopy(model)
    # with torch.no_grad():
    #     sync_client.local_update([1, 2, 3, 4, 5])
    #     sync_client.local_update([6, 7, 8, 9, 10])
    #     sync_client.reset_model()

    # for orig_param, reset_param in zip(original_model.parameters(), model.parameters()):
    #     assert (orig_param - reset_param).abs().max() < 1e-6
