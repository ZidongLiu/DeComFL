- optional, only when want to use layer-wise sparsity. generate_pruning_sparsity.py: initialization pruning. example usage: `python generate_pruning_sparsity.py --dataset=cifar10`

- ZOO random gradient estimate + SGD training. rge_main.py: train model using ZOO RGE. example usage:

  - When have default sparsity file: `python rge_main.py --dataset=cifar10 --sparsity-file=saved_sparsity/cifar10/zoo_grasp_0.9.json`
  - When do not intend to prune: `python rge_main.py --dataset=cifar10`

- CeZO-FL: Follow FL routine. And split data into chunks and train on different clients. example usage: `python cezo_fl_main.py --dataset=shakespeare --iterations=1000 --eval-iterations=200 --num-clients=5 --num-sample-clients=3 --local-update-steps=1 --num-pert=5 --grad-estimate-method=rge-forward`
