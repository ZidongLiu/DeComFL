from cezo_grpc import sample_pb2

def protobuf_to_py_list_of_ints(message: sample_pb2.ListOfInts) -> list[int]:
	return [v for v in message.data]

def protobuf_to_py_list_of_list_of_ints(message: sample_pb2.ListOfListOfInts) -> list[list[int]]:
	return [protobuf_to_py_list_of_ints(v) for v in message.data]

def protobuf_to_py_list_of_floats(message: sample_pb2.ListOfFloats) -> list[float]:
	return [v for v in message.data]

def protobuf_to_py_list_of_list_of_floats(message: sample_pb2.ListOfListOfFloats) -> list[list[float]]:
	return [protobuf_to_py_list_of_floats(v) for v in message.data]

def protobuf_to_py_list_of_list_of_list_of_floats(message: sample_pb2.ListOfListOfListOfFloats) -> list[list[list[float]]]:
	return [protobuf_to_py_list_of_list_of_floats(v) for v in message.data]
