#!/bin/bash

python run_grpc_server.py &
python run_fast_api.py &
wait