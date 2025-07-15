import torch
from experiment_helper.cli_parser import DeviceSetting
from cezo_fl.fl_helpers import get_client_name, get_server_name


def use_device(device_setting: DeviceSetting, num_clients: int) -> dict[str, torch.device]:
    use_cuda = device_setting.cuda and torch.cuda.is_available()
    use_mps = device_setting.mps and torch.backends.mps.is_available()
    if use_cuda:
        num_gpu = torch.cuda.device_count()
        print(f"----- Using cuda count: {num_gpu} -----")
        # num_workers will make dataloader very slow especially when number clients is large
        server_device = {get_server_name(): torch.device("cuda:0")}
        client_devices = {
            get_client_name(i): torch.device(f"cuda:{(i+1) % num_gpu}") for i in range(num_clients)
        }
    elif use_mps:
        print("----- Using mps -----")
        print("----- Model Dtype must be float32 -----")
        server_device = {get_server_name(): torch.device("mps")}
        client_devices = {get_client_name(i): torch.device("mps") for i in range(num_clients)}
    else:
        print("----- Using cpu -----")
        print("----- Model Dtype must be float32 -----")
        server_device = {get_server_name(): torch.device("cpu")}
        client_devices = {get_client_name(i): torch.device("cpu") for i in range(num_clients)}

    return server_device | client_devices
