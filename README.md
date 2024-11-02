# Dimension-Free Communication in Federated Learning (DeComFL)

![ci](https://github.com/ZidongLiu/FedDisco/actions/workflows/ci.yaml/badge.svg) ![apache](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

DeComFL is a library designed for training/fine-tuning deep learning models in the federated learning scenario. Its unique feature is the utilization of zeroth-order optimization, enabling communication between clients to be limited to just a few scalars, irrespective of the original model's size. This dimension-free communication is the inspiration behind the library's name.

## Performance

From Tables 1 and 2, we observe the DeComFL's effectiveness in communication cost reduction. We evaluate its performance with five and ten perturbations. Its performance matches or even outperforms MeZO and FedZO in all datasets. Surprisingly, DeComFL can just require about **1MB communication cost** to converge, which is a significant saving compared with other algorithms. 

<table>
  <caption style="caption-side: top; text-align: center; font-weight: bold;">Table 1: Test accuracy and communication cost on fine-tuning tasks</caption>
  <thead>
    <tr>
      <th style="text-align: center;">Model</th>
      <th style="text-align: center;">Dataset</th>
      <th style="text-align: center;">MeZO</th>
      <th style="text-align: center;">FedZO(P=5)</th>
      <th style="text-align: center;">DeComFL(P=5)</th>
      <th style="text-align: center;">DeComFL(P=10)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6" style="text-align: center;">OPT-125M</td>
      <td style="text-align: center;">SST-2</td>
      <td style="text-align: center;">84.0%</td>
      <td style="text-align: center;">84.1%(0.27 TB)</td>
      <td style="text-align: center;">84.0%(0.18 MB)</td>
      <td style="text-align: center;">85.1%(0.36 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">CB</td>
      <td style="text-align: center;">72.5%</td>
      <td style="text-align: center;">74.0%(0.09 TB)</td>
      <td style="text-align: center;">74.3%(0.06 MB)</td>
      <td style="text-align: center;">75.0%(0.12 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">WSC</td>
      <td style="text-align: center;">55.2%</td>
      <td style="text-align: center;">59.4%(0.27 TB)</td>
      <td style="text-align: center;">59.1%(0.18 MB)</td>
      <td style="text-align: center;">59.6%(0.36 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">WIC</td>
      <td style="text-align: center;">53.3%</td>
      <td style="text-align: center;">53.3%(0.27 TB)</td>
      <td style="text-align: center;">53.3%(0.18 MB)</td>
      <td style="text-align: center;">53.4%(0.36 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">RTE</td>
      <td style="text-align: center;">52.9%</td>
      <td style="text-align: center;">53.4%(0.18 TB)</td>
      <td style="text-align: center;">54.3%(0.12 MB)</td>
      <td style="text-align: center;">57.1%(0.24 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">BoolQ</td>
      <td style="text-align: center;">61.5%</td>
      <td style="text-align: center;">61.2%(0.18 TB)</td>
      <td style="text-align: center;">61.4%(0.12 MB)</td>
      <td style="text-align: center;">61.6%(0.24 MB)</td>
    </tr>
    <tr>
      <td rowspan="6" style="text-align: center;">OPT-1.3B</td>
      <td style="text-align: center;">SST-2</td>
      <td style="text-align: center;">90.2%</td>
      <td style="text-align: center;">90.2%(1937.2 TB)</td>
      <td style="text-align: center;">90.0%(0.12 MB)</td>
      <td style="text-align: center;">90.8%(0.24 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">CB</td>
      <td style="text-align: center;">74.0%</td>
      <td style="text-align: center;">74.4%(2905.7 TB)</td>
      <td style="text-align: center;">74.4%(0.18 MB)</td>
      <td style="text-align: center;">75.7%(0.36 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">WSC</td>
      <td style="text-align: center;">58.2%</td>
      <td style="text-align: center;">60.0%(2905.7 TB)</td>
      <td style="text-align: center;">60.4%(0.18 MB)</td>
      <td style="text-align: center;">64.2%(0.36 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">WIC</td>
      <td style="text-align: center;">56.0%</td>
      <td style="text-align: center;">56.1%(1937.2 TB)</td>
      <td style="text-align: center;">56.0%(0.12 MB)</td>
      <td style="text-align: center;">56.1%(0.24 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">RTE</td>
      <td style="text-align: center;">57.6%</td>
      <td style="text-align: center;">58.9%(1452.9 TB)</td>
      <td style="text-align: center;">59.4%(0.90 MB)</td>
      <td style="text-align: center;">60.9%(1.80 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">BoolQ</td>
      <td style="text-align: center;">62.0%</td>
      <td style="text-align: center;">62.0%(1452.9 TB)</td>
      <td style="text-align: center;">62.2%(0.90 MB)</td>
      <td style="text-align: center;">62.5%(1.80 MB)</td>
    </tr>
  </tbody>
</table>


<table>
  <caption style="caption-side: top; text-align: center; font-weight: bold;">Table 2: Test accuracy on fine-tuning tasks (LoRA)</caption>
  <thead>
    <tr>
      <th style="text-align: center;">Model</th>
      <th style="text-align: center;">Dataset</th>
      <th style="text-align: center;">MeZO</th>
      <th style="text-align: center;">FedZO(P=5)</th>
      <th style="text-align: center;">DeComFL(P=5)</th>
      <th style="text-align: center;">DeComFL(P=10)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;" rowspan="6">OPT-125M</td>
      <td style="text-align: center;">SST-2</td>
      <td style="text-align: center;">85.1%</td>
      <td style="text-align: center;">85.3% (279.40 TB)</td>
      <td style="text-align: center;">85.4% (0.18 MB)</td>
      <td style="text-align: center;">85.4% (0.36 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">CB</td>
      <td style="text-align: center;">69.6%</td>
      <td style="text-align: center;">70.6% (93.13 TB)</td>
      <td style="text-align: center;">71.1% (0.06 MB)</td>
      <td style="text-align: center;">71.4% (0.12 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">WSC</td>
      <td style="text-align: center;">52.7%</td>
      <td style="text-align: center;">54.6% (93.13 TB)</td>
      <td style="text-align: center;">54.5% (0.06 MB)</td>
      <td style="text-align: center;">57.0% (0.12 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">WIC</td>
      <td style="text-align: center;">53.5%</td>
      <td style="text-align: center;">53.1% (186.26 TB)</td>
      <td style="text-align: center;">53.1% (0.12 MB)</td>
      <td style="text-align: center;">53.7% (0.24 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">RTE</td>
      <td style="text-align: center;">50.2%</td>
      <td style="text-align: center;">50.9% (46.57 TB)</td>
      <td style="text-align: center;">51.4% (0.03 MB)</td>
      <td style="text-align: center;">51.4% (0.06 MB)</td>
    </tr>
    <tr>
      <td style="text-align: center;">BoolQ</td>
      <td style="text-align: center;">60.7%</td>
      <td style="text-align: center;">60.5% (46.57 TB)</td>
      <td style="text-align: center;">60.1% (0.03 MB)</td>
      <td style="text-align: center;">60.8% (0.06 MB)</td>
    </tr>
  </tbody>
</table>



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
