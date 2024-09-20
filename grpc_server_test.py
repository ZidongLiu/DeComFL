from grpc_server import SampleServer


def test_get_next_connect_client_index():
    server = SampleServer()
    server.connected_clients = [False, False, False]
    assert server._get_next_connect_client_index() == 0

    server.connected_clients = [True, False, False]
    assert server._get_next_connect_client_index() == 1

    server.connected_clients = [True, True, False]
    assert server._get_next_connect_client_index() == 2

    server.connected_clients = [False, True, False]
    assert server._get_next_connect_client_index() == 0

    server.connected_clients = [True, True, True]
    assert server._get_next_connect_client_index() == -1


def test_preprare_for_next_iteration():
    server = SampleServer()
    pass
