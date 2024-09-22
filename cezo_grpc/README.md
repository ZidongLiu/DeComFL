run this at root `python -m grpc_tools.protoc -Icezo_grpc=./cezo_grpc --python_out=. --grpc_python_out=.  ./cezo_grpc/sample.proto`

exaple to run large llm `python run_grpc.py --dataset=sst2 --eval-iterations=25 --large-model=opt-125m --model-dtype=float16 --seed=365 --iterations=2000 --train-batch-size=32 --test-batch-size=64 --num-clients=3 --num-sample-clients=2 --local-update-steps=1 --num-pert=5 --lr=0.000005 --momentum=0 --grad-estimate-method=rge-forward --mu=0.001`
