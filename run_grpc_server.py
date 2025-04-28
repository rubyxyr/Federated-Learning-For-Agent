from server.server import BaseServer


def serve():
    server = BaseServer("config.yaml")
    server.run_grpc_server()


if __name__ == '__main__':
    serve()
