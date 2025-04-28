import argparse
from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from client.client import BaseClient
from server.server import BaseServer
from omegaconf import OmegaConf
from threading import Thread
from server.strategy.fedavg import FedAvg
from server.strategy.dp_fixed_clip import DpServerFixedClip


CFG_PATH = "config.yaml"
config_detail = OmegaConf.load(CFG_PATH)


def runserver_with_dp():
    dp_strategy = DpServerFixedClip(
        cfg_path=CFG_PATH,
        strategy=FedAvg()
    )
    server = BaseServer(cfg_path=CFG_PATH, strategy=dp_strategy)
    server.run_grpc_server()


def runserver():
    server = BaseServer(CFG_PATH)
    server.run_grpc_server()


def run_simulation(use_server_dp=False):
    server_side = runserver if not use_server_dp else runserver_with_dp
    server_thread = Thread(target=server_side)
    server_thread.start()

    client = BaseClient("123355", CFG_PATH)
    client.run_grpc_client()

    server_thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run local FL simulation"
    )
    parser.add_argument("--use_server_dp", required=False, type=bool, default=False)
    args = parser.parse_args()

    if args.use_server_dp:
        run_simulation(True)
    else:
        run_simulation()


