import argparse
from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from client.client import BaseClient
from omegaconf import OmegaConf


CFG_PATH = "config.yaml"
config_detail = OmegaConf.load(CFG_PATH)


def update(client_id):
    # update local model by merging latest weight file from server side to local model
    client = BaseClient(client_id, CFG_PATH)
    client.update()


def run(client_id):
    # run client train and send weight to server side
    client = BaseClient(client_id, CFG_PATH)
    client.run_grpc_client()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run local client"
    )
    parser.add_argument("--client_id", required=False, type=str, default='12366')
    parser.add_argument("--local_update", required=False, type=bool, default=False, help="update local model by merging latest weight file from server side to local model")
    args = parser.parse_args()

    if args.local_update:
        update(args.client_id)
    else:
        run(args.client_id)


