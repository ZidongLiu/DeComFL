- optional, only when want to use layer-wise sparsity. generate_pruning_sparsity.py: initialization pruning. example usage: `python generate_pruning_sparsity.py --dataset=cifar10`

- ZOO random gradient estimate + SGD training. rge_main.py: train model using ZOO RGE. example usage:

  - When have default sparsity file: `python rge_main.py --dataset=cifar10 --sparsity-file=saved_sparsity/cifar10/zoo_grasp_0.9.json`
  - When do not intend to prune: `python rge_main.py --dataset=cifar10`

- CeZO-FL: Follow FL routine. And split data into chunks and train on different clients. example usage: `python cezo_fl_main.py --dataset=sst2 --iterations=1000 --train-batch-size=8 --test-batch-size=200 --eval-iterations=50 --num-clients=3 --num-sample-clients=2 --local-update-steps=1 --num-pert=10 --lr=1e-6 --mu=1e-3 --grad-estimate-method=rge-forward`
