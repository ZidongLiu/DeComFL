# Dimension-Free Communication in Federated Learning (DeComFL)

![ci](https://github.com/ZidongLiu/FedDisco/actions/workflows/ci.yaml/badge.svg) ![apache](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

DeComFL is a library designed for training/fine-tuning deep learning models in the federated learning scenario. Its unique feature is the utilization of zeroth-order optimization, enabling communication between clients to be limited to just a few scalars, irrespective of the original model's size. This dimension-free communication is the inspiration behind the library's name.

## Environment Setup

We use [conda](https://docs.conda.io/projects/conda/en/stable/) as our cross-platform environment management tool. However, due to macOS' lacking support for cuda, we have to make 2 different environment setup files:

- Use `environment.yml` on macOS or if you do not have cuda at hand.
- Use `environment_cuda.yml` otherwise.

For READMD.md, we will use `environment.yml` whenever an environment file is needed.

### Set Up Steps

1. Make sure `conda` is available. See https://conda.io/projects/conda/en/latest/user-guide/install/index.html for more detail.
2. At the root of this repo, run `conda env create -f environment.yml -y`.
3. Once installation is finished, run `conda activate decomfl` to use the created virtual env.
4. (Optional) If you see something like `conda init before activate`. Run `conda init`, then restart your terminal/powershell. Then repeat step 3.
5. Run any command provided in [Run Experiments](#run-experiments) section. If code works, then congratulations, you have successfully set up the environment for this repo!
6. Update the environemtn if there are some missing dependencies, most recent change was introduced by adding grpc. Try `conda env update --file environment.yml --prune`. The `--prune` is optional, if `--prune` conda will remove any dependencies that are no longer required from the environment.

## Run Experiments

- **Run zeroth-order random gradient estimate + SGD training**. Train model using ZOO RGE.
  Usage example: `python zo_rge_main.py --dataset=mnist --num-pert=10 --lr=1e-5 --mu=1e-3 --momentum=0.9`

- **Run DeComFL:** Follow FL routine, split data into chunks and train on different clients.
  Usage example: `python decomfl_main.py --large-model=opt-125m --dataset=sst2 --iterations=1000 --train-batch-size=32 --test-batch-size=200 --eval-iterations=25 --num-clients=3 --num-sample-clients=2 --local-update-steps=1 --num-pert=5 --lr=1e-5 --mu=1e-3 --grad-estimate-method=rge-forward`

- **Run FedAvg:** Run standard fedavg algorithm.
  `python fo_fl_main.py --dataset=sst2 --lr=1e-3 --num-clients=5 --num-sample-clients=3 --local-update-steps=1 --train-batch-size=32 --test-batch-size=200 --momentum=0.9`

## Citation

```
@article{li2024achieving,
  title={Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization},
  author={Li, Zhe and Ying, Bicheng and Liu, Zidong and Dong, Chaosheng and Yang, Haibo},
  journal={arXiv preprint arXiv:2405.15861},
  year={2024}
}
```

## Contributors

DeComFL is currently contributed and maintained by <a href="https://zidongliu.github.io/" style="text-decoration: none;">**Zidong Liu**</a> (ComboCurve), Dr. <a href="https://scholar.google.com/citations?user=LuF6KX4AAAAJ&hl=en&oi=ao" style="text-decoration: none;">**Bicheng Ying**</a> (Google) and <a href="https://rogerrogerusc.github.io/" style="text-decoration: none;">**Zhe Li**</a> (RIT), and advised by Dr. <a href="https://haibo-yang-osu.github.io/homepage/" style="text-decoration: none;">**Haibo Yang**</a> (RIT).

<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/user-attachments/assets/b3982917-e302-42c3-b396-e33bb9f52c90" alt="Image 1" style="width: 80%;" />
    <div style="display: flex; justify-content: center;">
      <img src="https://github.com/user-attachments/assets/c0dfb199-0a51-4b17-b9ba-9fe09d2c4f7a" alt="Image 2" style="width: 51%;" /> &nbsp;&nbsp;&nbsp;&nbsp;
      <img src="https://github.com/user-attachments/assets/23ba00dc-fc62-4ab3-9c70-0326aa20b786" alt="Image 3" style="width: 25%;" />
    </div>
</div>
