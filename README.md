# Dimension-Free Communication in Federated Learning (DeComFL)

![ci](https://github.com/ZidongLiu/FedDisco/actions/workflows/ci.yaml/badge.svg) ![apache](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

DeComFL is a library designed for training/fine-tuning deep learning models in the federated learning scenario. Its unique feature is the utilization of zeroth-order optimization, enabling communication between clients to be limited to just a few scalars, irrespective of the original model's size. This dimension-free communication is the inspiration behind the library's name.

## Performance

From Table 1 and 2, we observe the DeComFL's effectiveness in communication cost reduction. We evaluate its performance with five and ten perturbations. Its performance matches or even outperforms MeZO and FedZO in all datasets. Surprisingly, DeComFL can just require about **1MB communication cost** to converge, which is a significant saving compared with other algorithms. 

<table>
  <caption style="caption-side: top; text-align: center; font-weight: bold;">Table 1: Test accuracy and communication cost on fine-tuning tasks</caption>
  <thead>
    <tr>
      <th style="text-align: center;">Model</th>
      <th style="text-align: center;">Dataset / Task</th>
      <th style="text-align: center;">MeZO</th>
      <th style="text-align: center;">FedZO with P = 5</th>
      <th style="text-align: center;">DeComFL with P = 5</th>
      <th style="text-align: center;">DeComFL with P = 10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6" style="text-align: center;">OPT-125M</td>
      <td style="text-align: center;">SST-2</td>
      <td style="text-align: center;">83.99%</td>
      <td style="text-align: center;">84.11% (0.27 TB)</td>
      <td style="text-align: center;">84.02% (0.18 MB)</td>
      <td style="text-align: center;">85.08% (0.36 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">CB</td>
      <td style="text-align: center;">72.49%</td>
      <td style="text-align: center;">73.97% (0.09 TB)</td>
      <td style="text-align: center;">74.28% (0.06 MB)</td>
      <td style="text-align: center;">75.00% (0.12 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">WSC</td>
      <td style="text-align: center;">55.18%</td>
      <td style="text-align: center;">59.43% (0.27 TB)</td>
      <td style="text-align: center;">59.13% (0.18 MB)</td>
      <td style="text-align: center;">59.59% (0.36 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">WIC</td>
      <td style="text-align: center;">53.25%</td>
      <td style="text-align: center;">53.31% (0.27 TB)</td>
      <td style="text-align: center;">53.28% (0.18 MB)</td>
      <td style="text-align: center;">53.38% (0.36 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">RTE</td>
      <td style="text-align: center;">52.91%</td>
      <td style="text-align: center;">53.42% (0.18 TB)</td>
      <td style="text-align: center;">54.33% (0.12 MB)</td>
      <td style="text-align: center;">57.05% (0.24 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">BoolQ</td>
      <td style="text-align: center;">61.46%</td>
      <td style="text-align: center;">61.20% (0.18 TB)</td>
      <td style="text-align: center;">61.36% (0.12 MB)</td>
      <td style="text-align: center;">61.60% (0.24 MB)</td>
    </tr>
    <tr>
      <td rowspan="6" style="text-align: center;">OPT-1.3B</td>
      <td style="text-align: center;">SST-2</td>
      <td style="text-align: center;">90.23%</td>
      <td style="text-align: center;">90.17% (1937.15 TB)</td>
      <td style="text-align: center;">90.02% (0.12 MB)</td>
      <td style="text-align: center;">90.78% (0.24 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">CB</td>
      <td style="text-align: center;">74.01%</td>
      <td style="text-align: center;">74.41% (2905.73 TB)</td>
      <td style="text-align: center;">74.40% (0.18 MB)</td>
      <td style="text-align: center;">75.71% (0.36 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">WSC</td>
      <td style="text-align: center;">58.21%</td>
      <td style="text-align: center;">59.95% (2905.73 TB)</td>
      <td style="text-align: center;">60.41% (0.18 MB)</td>
      <td style="text-align: center;">64.16% (0.36 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">WIC</td>
      <td style="text-align: center;">55.95%</td>
      <td style="text-align: center;">56.06% (1937.15 TB)</td>
      <td style="text-align: center;">55.97% (0.12 MB)</td>
      <td style="text-align: center;">56.14% (0.24 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">RTE</td>
      <td style="text-align: center;">57.57%</td>
      <td style="text-align: center;">58.88% (1452.86 TB)</td>
      <td style="text-align: center;">59.42% (0.90 MB)</td>
      <td style="text-align: center;">60.89% (1.80 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">BoolQ</td>
      <td style="text-align: center;">61.98%</td>
      <td style="text-align: center;">62.01% (1452.86 TB)</td>
      <td style="text-align: center;">62.17% (0.90 MB)</td>
      <td style="text-align: center;">62.50% (1.80 MB)</td>
    </tr>
  </tbody>
</table>


<table>
  <caption style="caption-side: top; text-align: center; font-weight: bold;">Table 2: Test accuracy on fine-tuning tasks (LoRA)</caption>
  <thead>
    <tr>
      <th style="text-align: center;">Model</th>
      <th style="text-align: center;">Dataset / Task</th>
      <th style="text-align: center;">MeZO</th>
      <th style="text-align: center;">FedZO with P = 5</th>
      <th style="text-align: center;">DeComFL with P = 5</th>
      <th style="text-align: center;">DeComFL with P = 10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;" rowspan="6">OPT-125M</td>
      <td style="text-align: center;">SST-2</td>
      <td style="text-align: center;">85.07%</td>
      <td style="text-align: center;">85.34% (279.40 TB)</td>
      <td style="text-align: center;">85.42% (0.18 MB)</td>
      <td style="text-align: center;">85.44% (0.36 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">CB</td>
      <td style="text-align: center;">69.64%</td>
      <td style="text-align: center;">70.55% (93.13 TB)</td>
      <td style="text-align: center;">71.07% (0.06 MB)</td>
      <td style="text-align: center;">71.43% (0.12 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">WSC</td>
      <td style="text-align: center;">52.66%</td>
      <td style="text-align: center;">54.61% (93.13 TB)</td>
      <td style="text-align: center;">54.53% (0.06 MB)</td>
      <td style="text-align: center;">57.03% (0.12 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">WIC</td>
      <td style="text-align: center;">53.49%</td>
      <td style="text-align: center;">53.12% (186.26 TB)</td>
      <td style="text-align: center;">53.08% (0.12 MB)</td>
      <td style="text-align: center;">53.71% (0.24 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">RTE</td>
      <td style="text-align: center;">50.15%</td>
      <td style="text-align: center;">50.92% (46.57 TB)</td>
      <td style="text-align: center;">51.40% (0.03 MB)</td>
      <td style="text-align: center;">51.40% (0.06 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">BoolQ</td>
      <td style="text-align: center;">60.68%</td>
      <td style="text-align: center;">60.53% (46.57 TB)</td>
      <td style="text-align: center;">60.12% (0.03 MB)</td>
      <td style="text-align: center;">60.78% (0.06 MB)</td>
    </tr>
  </tbody>
</table>



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
6. Update the environemtn if there are some missing dependencies, most recent change was introduced by adding grpc. Try `conda env update --file environment.yml --prune`. The `--prune` option causes conda to remove any dependencies that are no longer required from the environment.

## Run Experiments

- **Run zeroth-order random gradient estimate + SGD training**. Train model using ZOO RGE.
  Usage example: `python zo_rge_main.py --dataset=cifar10 --num-pert=10 --lr=1e-6 --mu=1e-3`

- **Run DeComFL:** Follow FL routine, split data into chunks and train on different clients.
  Usage example: `python decomfl_main.py --large-model=opt-125m --dataset=sst2 --iterations=1000 --train-batch-size=32 --test-batch-size=200 --eval-iterations=25 --num-clients=3 --num-sample-clients=2 --local-update-steps=1 --num-pert=5 --lr=1e-5 --mu=1e-3 --grad-estimate-method=rge-forward`

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
DeComFL is currently contributed and maintained by <a href="https://zidongliu.github.io/" style="text-decoration: none;">**Zidong Liu**</a> (ComboCurve), <a href="https://scholar.google.com/citations?user=LuF6KX4AAAAJ&hl=en&oi=ao" style="text-decoration: none;">**Bicheng Ying**</a> (Google) and <a href="https://rogerrogerusc.github.io/" style="text-decoration: none;">**Zhe Li**</a> (RIT), and advised by Prof. <a href="https://haibo-yang-osu.github.io/homepage/" style="text-decoration: none;">**Haibo Yang**</a> (RIT). 

<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/user-attachments/assets/b3982917-e302-42c3-b396-e33bb9f52c90" alt="Image 1" style="width: 80%;" />
    <div style="display: flex; justify-content: center;">
      <img src="https://github.com/user-attachments/assets/c0dfb199-0a51-4b17-b9ba-9fe09d2c4f7a" alt="Image 2" style="width: 51%;" /> &nbsp;&nbsp;&nbsp;&nbsp;
      <img src="https://github.com/user-attachments/assets/23ba00dc-fc62-4ab3-9c70-0326aa20b786" alt="Image 3" style="width: 25%;" />
    </div>
</div>
