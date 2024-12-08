from cezo_grpc import sample_pb2


def protobuf_to_py_list_of_ints(message: sample_pb2.ListOfInts) -> list[int]:
    return [v for v in message.data]


def protobuf_to_py_list_of_list_of_ints(message: sample_pb2.ListOfListOfInts) -> list[list[int]]:
    return [protobuf_to_py_list_of_ints(v) for v in message.data]


def protobuf_to_py_list_of_floats(message: sample_pb2.ListOfFloats) -> list[float]:
    return [v for v in message.data]


def protobuf_to_py_list_of_list_of_floats(
    message: sample_pb2.ListOfListOfFloats,
) -> list[list[float]]:
    return [protobuf_to_py_list_of_floats(v) for v in message.data]


def protobuf_to_py_list_of_list_of_list_of_floats(
    message: sample_pb2.ListOfListOfListOfFloats,
) -> list[list[list[float]]]:
    return [protobuf_to_py_list_of_list_of_floats(v) for v in message.data]


def py_to_protobuf_list_of_ints(py_data: list[int]) -> sample_pb2.ListOfInts:
    return sample_pb2.ListOfInts(data=py_data)


def py_to_protobuf_list_of_list_of_ints(py_data: list[list[int]]) -> sample_pb2.ListOfListOfInts:
    return sample_pb2.ListOfListOfInts(data=[py_to_protobuf_list_of_ints(v) for v in py_data])


def py_to_protobuf_list_of_floats(py_data: list[float]) -> sample_pb2.ListOfFloats:
    return sample_pb2.ListOfFloats(data=py_data)


def py_to_protobuf_list_of_list_of_floats(
    py_data: list[list[float]],
) -> sample_pb2.ListOfListOfFloats:
    return sample_pb2.ListOfListOfFloats(data=[py_to_protobuf_list_of_floats(v) for v in py_data])


def py_to_protobuf_list_of_list_of_list_of_floats(
    py_data: list[list[list[float]]],
) -> sample_pb2.ListOfListOfListOfFloats:
    return sample_pb2.ListOfListOfListOfFloats(
        data=[py_to_protobuf_list_of_list_of_floats(v) for v in py_data]
    )
