import grpc
from cezo_grpc import sample_pb2
from cezo_grpc import sample_pb2_grpc
from cezo_grpc import data_helper

from cezo_fl import client

rpc_master_addr = "localhost"
rpc_master_port = 4242
channel = grpc.insecure_channel(f"{rpc_master_addr}:{rpc_master_port}")
ps_stub = sample_pb2_grpc.SampleServerStub(channel)

connect_result = ps_stub.Connect(sample_pb2.EmptyRequest())

successful, client_index = connect_result.successful, connect_result.clientIndex
print(successful, client_index)

join_result = ps_stub.TryToJoinIteration(sample_pb2.TryToJoinIterationRequest(clientIndex = client_index))
print(join_result.successful)

grads_and_seeds = ps_stub.PullGradsAndSeeds(sample_pb2.PullGradsAndSeedsRequest(clientIndex = client_index))

print(data_helper.protobuf_to_py_list_of_list_of_list_of_floats(grads_and_seeds.grads), data_helper.protobuf_to_py_list_of_list_of_ints(grads_and_seeds.seeds))

ps_stub.SubmitIteration(sample_pb2.SubmitIterationRequest(local_update_result = sample_pb2.ListOfListOfFloats(data=[sample_pb2.ListOfFloats(data=[0.123])])))

print('success')