import grpc
import data_helper
from concurrent import futures
import sample_pb2_grpc
import sample_pb2


class SampleServer(sample_pb2_grpc.SampleServerServicer):
    def Connect(self, request, context):
        return sample_pb2.ConnectResponse(
            successful = True,
            clientIndex = 0
        )
    
    def TryToJoinIteration(self, request, context):
        return sample_pb2.TryToJoinIterationResponse(
            successful = True
        )
    
    def PullGradsAndSeeds(self, request, context):
        seeds = sample_pb2.ListOfListOfInts(data=[sample_pb2.ListOfInts(data=[1])])
        grads = sample_pb2.ListOfListOfListOfFloats(data=[sample_pb2.ListOfListOfFloats(data=[sample_pb2.ListOfFloats(data=[0.3])])])
        return sample_pb2.PullGradsAndSeedsResponse(seeds=seeds, grads=grads)

    def SubmitIteration(self, request, context):
        print(data_helper.protobuf_to_py_list_of_list_of_floats(request.local_update_result))
        return sample_pb2.EmptyResponse()

def serve(rpc_master_port, rpc_num_workers):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=rpc_num_workers))
    sample_pb2_grpc.add_SampleServerServicer_to_server(SampleServer(), server)
    server.add_insecure_port(f"localhost:{rpc_master_port}")
    print(f"Parameter server starting on [::]:{rpc_master_port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve(4242, 8)
