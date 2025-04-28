# Federated Learning For LLM Agent

## Setup

Clone the project and prepare the local environment:
```
conda create -n flagent python=3.11
conda activate flagent
pip install -r requirements.txt
```

## Configuring Files
In our initial **config.yaml** file, you need to set your **model path** and **dataset name**. 

```
model:
  model_path:  # model path in Hugging face or local model path
  quantization: 8 # if you want to use cpu, please set this to null
  device_map: "cuda" # support cuda, cpu and mps
 
dataset_name： # dataset in Hugging face or local dataset path
```


## Local Federated Learning Fine-tuning

**Efficiency**: We consider the use of Parameter-Efficient Fine-Tuning for local clients, such as LoRA. 

To start the framework test on a machine and simulate the federated learning fine-tuning process, run the following command:

```
python main_fl_socket.py
```

The client side uses [local differential privacy](https://en.wikipedia.org/wiki/Local_differential_privacy) by default, you can close it by setting **local_dp** in client block to **False** in `config.yaml`

If you want to use differential privacy in server side, you can set **local_dp** in client block to **False** in `config.yaml` and run:
 ```
 python main_fl_socket.py --use_server_dp=true
 ```

It also supports [gRPC](https://grpc.io/) for client and server communication, you can run this script to simulate:

```
python main_fl_grpc.py
```

If you want to use differential privacy in server side, you can set **local_dp** in client block to **False** in `config.yaml` and run:
 ```
 python main_fl_grpc.py --use_server_dp=true
 ```

> It supports to create an insecure and secure gRPC channel, you can set your local [root certificates](https://en.wikipedia.org/wiki/Root_certificate) to config.yaml to use secure channel:
>  ```
> client:
>   grpc_insecure: True # you can set it to False to turn off it and set grpc_auth_cer_path to use secure gRPC channel
>   grpc_auth_cer_path: null # set your local root certificates path to here
> ```

You can modify [proto file](https://protobuf.dev/getting-started/pythontutorial/) `utils/protos/communicate.proto` to generate your new message structure and communication function.

Now in server side, we have only two strategies under `server/strategy/`, we will add more in the future. 
- [x] Federate average (default strategy in server side)
> The central server aggregates the models received from the clients by averaging the model parameters. 
- [x] Federate average + differential privacy with fixed clipping
> When implementing differential privacy, data often needs to be processed to reduce sensitivity. Fixed clipping is one such method. It refers to clipping or limiting the data according to a preset threshold before it is used for further analysis. The purpose of this is to control the range of the data, thereby reducing the impact of individual extreme values ​​on the final result and ensuring that the output of the algorithm does not pose a risk to the privacy of any individual.
- [ ] Federate average + differential privacy with adaptive clipping
> Different from fixed clipping, adaptive clipping does not pre-set a fixed clipping threshold, but dynamically adjusts the threshold according to the actual distribution of data and the required privacy protection level.
- [ ] ...

### Evaluation
You can use `utils/eval_from_local.py` as a script to evaluate the model.
```
cd utils
python eval_from_local.py
```
You can customize the script by setting parameters, running script with -h to see each parameter description:
```commandline
python eval_from_local.py -h

usage: eval_from_local.py [-h] [--ntrain NTRAIN] [--selected_subjects SELECTED_SUBJECTS] [--save_dir SAVE_DIR] [--lora_config_path LORA_CONFIG_PATH]
                          [--lora_weights_path LORA_WEIGHTS_PATH] [--global_record_file GLOBAL_RECORD_FILE] [--model MODEL]

options:
  -h, --help            show this help message and exit
  --ntrain NTRAIN, -k NTRAIN
                        few-shot examples amount, default is 3
  --selected_subjects SELECTED_SUBJECTS, -sub SELECTED_SUBJECTS
                        selected subjects: biology, business, chemistry, computer science, economics, engineering, health, history, law, math, philosophy, physics, psychology, other,
                        all. default is 'all'
  --save_dir SAVE_DIR, -s SAVE_DIR
                        evaluation results save dir, default is 'eval_results'
  --lora_config_path LORA_CONFIG_PATH, -lc LORA_CONFIG_PATH
                        lora config folder path
  --lora_weights_path LORA_WEIGHTS_PATH, -lw LORA_WEIGHTS_PATH
                        lora weights bin file path
  --global_record_file GLOBAL_RECORD_FILE, -grf GLOBAL_RECORD_FILE
                        global log record file, default is 'eval_record_collection.csv'
  --model MODEL, -m MODEL
                        local model path
```
It uses [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) as our dataset to do evaluation.

Also it uses evaluation in server side `server/server.py`：
```python
def eval(self, lora_config_path='', model_weights_path='', clients_data_detail=None):
    results = eval_model(self.config_detail.model.model_path, lora_config_path, model_weights_path, n_train=1)
    result_save_path = os.path.join(self.output, self.latest_version)
    with open(result_save_path + '/eval_result.json', 'w') as f:
        json.dump(results, f)
    print(f'Eval results: {results}')


def update(self, do_eval=True):
    """Aggregate model and save new model weight, send reward to each client"""
    clients_detail, dataset_length_list, path_list = get_clients_uploads_after(self.save_path, self.latest_version)
    client_list = clients_detail.keys()
    if len(client_list) > 0:
        self.aggregate(clients_detail.keys(), dataset_length_list, path_list)
        self.save_model()
        if do_eval:
            weight_saved_path = os.path.join(self.output, self.latest_version, 'adapter_model.bin')
            eval_thread = Thread(target=self.eval,
                                 args=['./output',
                                       weight_saved_path,
                                       clients_detail])
            eval_thread.start()
    else:
        logging.info('No new weights from client')
```
