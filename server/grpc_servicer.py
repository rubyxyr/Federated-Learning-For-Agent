from utils.proto_py import communicate_pb2_grpc, communicate_pb2
from google.protobuf.json_format import MessageToDict
import os
import io
import torch
import json
from datetime import datetime


def deserialize_model_state_dict(serialized_state_dict):
    state_dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for key, byte_tensor in serialized_state_dict.items():
        # Use a BytesIO object to load the serialized tensor back into a tensor
        buffer = io.BytesIO(byte_tensor)
        tensor = torch.load(buffer, weights_only=True, map_location=torch.device(device))
        state_dict[key] = tensor
    return state_dict


def parse_value(value_pb):
    if value_pb.HasField('null_value'):
        return None
    elif value_pb.HasField('bool_value'):
        return value_pb.bool_value
    elif value_pb.HasField('number_value'):
        if value_pb.number_value.is_integer():
            return int(value_pb.number_value)
        return value_pb.number_value
    elif value_pb.HasField('string_value'):
        return value_pb.string_value
    elif value_pb.HasField('list_value'):
        return [parse_value(v) for v in value_pb.list_value.values]
    elif value_pb.HasField('struct_value'):
        if 'empty' in value_pb.struct_value.fields.keys():
            return {}
        return {k: parse_value(v) for k, v in value_pb.struct_value.fields.items()}
    else:
        raise TypeError(f"Unsupported google.protobuf.Value type: {value_pb}")


def protobuf_to_dict_with_none(protobuf_obj):
    dict_representation = MessageToDict(protobuf_obj, preserving_proto_field_name=True)

    descriptor = protobuf_obj.DESCRIPTOR
    ordered_data = {}
    for field in descriptor.fields:
        field_name = field.name
        if field_name in dict_representation:
            ordered_data[field_name] = dict_representation[field_name]
        else:
            ordered_data[field_name] = None

    return ordered_data


class WeightsTransferServicer(communicate_pb2_grpc.WeightsTransferServicer):
    def __init__(self, base_server):
        self.base_server = base_server

    def SendWeights(self, request, context):
        if request.HasField('send_parameters'):
            client_id = request.send_parameters.client_id
            train_dataset_length = request.send_parameters.train_dataset_length
            new_model_weight = request.send_parameters.new_model_weight
            lora_config = request.send_parameters.lora_config

            # Process the received weights here
            print(f"Received weights from client {client_id} with dataset length {train_dataset_length}")
            lora_data = dict()
            for lora_config in lora_config:
                lora_data[lora_config.config_name] = parse_value(lora_config.config_value)
            current_date = datetime.today().strftime("%Y%m%d_%H%M%S")
            client_save_path = os.path.join(
                self.base_server.save_path,
                "local_output_{}".format(str(client_id)),
                current_date.split('_')[0],
                current_date.split('_')[1],
            )
            os.makedirs(client_save_path, exist_ok=True)
            torch.save(
                deserialize_model_state_dict(new_model_weight),
                client_save_path + "/pytorch_model.bin",
            )
            with open(client_save_path + '/train_dataset_length.json', 'w') as f:
                json.dump({"train_dataset_length": train_dataset_length}, f)
            with open(client_save_path + '/adapter_config.json', 'w') as f:
                json.dump(lora_data, f)
            # Return a successful response
            return communicate_pb2.TransferStatus(code=True, message="Weights received successfully")
        else:
            return communicate_pb2.TransferStatus(code=False, message="Invalid message type")
