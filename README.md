# Dimension-Free Communication in Federated Learning (DeComFL)

![ci](https://github.com/ZidongLiu/FedDisco/actions/workflows/ci.yaml/badge.svg) ![apache](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

DeComFL is a library designed for training/fine-tuning deep learning models in the federated learning scenario. Its unique feature is the utilization of zeroth-order optimization, enabling communication between clients to be limited to just a few scalars, irrespective of the original model's size. This dimension-free communication is the inspiration behind the library's name.

## Performance

Table 1: Test accuracy comparsion of First-order SGD, MeZO and DeComFL on OPT-125m model.
|Dataset\Algo.| FO-SGD | MeZO | DeComFL (P=5) | DeComFL (P=10) |
|-|-|-|-|-|
|SST-2|87.48%|83.99%|84.02%|85.08%|
|CB|73.21%|72.49%|74.28%|75.00%|
|WSC|58.13%|55.18%|59.13%|59.59%|
|WIC|54.10%|53.25%|53.28%|53.38%|
|MultiRC|60.77%|58.36%|59.06%|60.39%|
|RTE|57.69%|52.91%|54.33%|57.05%|
|BoolQ|62.34%|61.46%|61.36%|61.60%|

From the Table 1, we observe the effectiveness of DeComFL. We evaluate its performance with five and ten perturbations. Its performance outperforms MeZO in almost all datasets, and it can match the performance of FO-SGD and even excel that sometimes (i.e., on CB and WSC datasets).

## Environment Setup

We use [conda](https://docs.conda.io/projects/conda/en/stable/) as our cross platform environment management tool. However due to macOS' lacking support for cuda, we have to make 2 different environment set up files:

- Use `environment.yml` on macOS or if you do not have cuda at hand.
- Use `environment_cuda.yml` otherwise.

For READMD.md, we will use `environment.yml` whenever a environment file is needed.

### Set Up Steps

1. Make sure `conda` is available, see https://conda.io/projects/conda/en/latest/user-guide/install/index.html for more detail.
2. At the root of this repo, run `conda env create -f environment.yml -y`.
3. Once installation is finished, run `conda activate decomfl` to use the created virtual env.
4. (Optional) If you see something like `conda init before activate`. Run `conda init`, then restart your terminal/powershell. Then repeat step 3.
5. Run any command provided in [Run Experiments](#run-experiments) section. If code works, then congratulations, you have successfully set up the environment for this repo!

## Run Experiments

- **Run zeroth-order random gradient estimate + SGD training**. Train model using ZOO RGE.
  Usage example: `python zo_rge_main.py --dataset=cifar10 --num-pert=10 --lr=1e-6 --mu=1e-3`

- **Run DeComFL:** Follow FL routine, split data into chunks and train on different clients.
  Usage example: `python decomfl_main.py --dataset=sst2 --iterations=10000 --train-batch-size=8 --test-batch-size=200 --eval-iterations=50 --num-clients=3 --num-sample-clients=2 --local-update-steps=1 --num-pert=10 --lr=1e-6 --mu=1e-3 --grad-estimate-method=rge-forward`

## Our Team
DeComFL is currently contributed and maintained by **Zidong Liu** (ComboCurve), **Bicheng Ying** (Google) and **Zhe Li** (RIT), and advised by Prof. **Haibo Yang** (RIT). 

<img src="https://github.com/user-attachments/assets/9e1062cf-7fd6-41ce-89f4-d23727cc9d45" alt="rit" width="550" style="float:left; padding-right:30px" />

<img src="https://github.com/user-attachments/assets/c0dfb199-0a51-4b17-b9ba-9fe09d2c4f7a" alt="combocurve" width="360" style="float:left; padding-right:30px" /><img src="https://github.com/user-attachments/assets/23ba00dc-fc62-4ab3-9c70-0326aa20b786" alt="google" width="150"/>


## Citation

```
@article{li2024achieving,
  title={Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization},
  author={Li, Zhe and Ying, Bicheng and Liu, Zidong and Yang, Haibo},
  journal={arXiv preprint arXiv:2405.15861},
  year={2024}
}
```
