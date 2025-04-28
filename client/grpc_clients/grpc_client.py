from contextlib import contextmanager
from pathlib import Path
import torch
import io
import logging
from logging import DEBUG
from google.protobuf import struct_pb2
from utils.grpc import create_channel, GRPC_MAX_MESSAGE_LENGTH
from .message import ClientSideMessage, SEND_PARAMETERS
from utils.proto_py import communicate_pb2, communicate_pb2_grpc


def serialize_model_state_dict(state_dict):
    serialized_state_dict = {}
    for key, tensor in state_dict.items():
        # Use a buffer to store the serialized tensor
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        serialized_state_dict[key] = buffer.getvalue()
    return serialized_state_dict


def convert_to_value(value):
    value_pb = struct_pb2.Value()
    if value is None:
        value_pb.null_value = struct_pb2.NULL_VALUE
    elif isinstance(value, bool):
        value_pb.bool_value = value
    elif isinstance(value, (int, float)):
        value_pb.number_value = value
    elif isinstance(value, str):
        value_pb.string_value = value
    elif isinstance(value, dict):
        if not value:  # Handle empty dictionary
            struct_value = value_pb.struct_value
            struct_value.update({"empty": True})
        else:
            struct_value = value_pb.struct_value
            struct_value.update({k: convert_to_value(v) for k, v in value.items()})
    elif isinstance(value, list):
        if not value:  # Handle empty list
            value_pb.list_value
        else:
            list_value = value_pb.list_value
            list_value.values.extend([convert_to_value(v) for v in value])
    else:
        raise TypeError(f"Unsupported type: {type(value)}")
    return value_pb


@contextmanager
def grpc_connection(server_address,
                    insecure,
                    root_certificates=None,
                    max_message_length=GRPC_MAX_MESSAGE_LENGTH):
    """Establish a gRPC connection to a gRPC server"""
    if isinstance(root_certificates, str):
        root_certificates = Path(root_certificates).read_bytes()

    channel = create_channel(
        server_address=server_address,
        insecure=insecure,
        root_certificates=root_certificates,
        max_message_length=max_message_length,
    )
    stub = communicate_pb2_grpc.WeightsTransferStub(channel)

    def receive():
        pass

    def send(message_data: ClientSideMessage):
        detail = message_data.content
        message_type = message_data.metadata.message_type

        if message_type == SEND_PARAMETERS:
            lora_config_message = [
                communicate_pb2.LoraConfig(
                    config_name=k,
                    config_value=convert_to_value(v)
                )
                for k, v in detail['lora_config'].items()
            ]
            msg_proto = communicate_pb2.ClientGrpcMessage(
                send_parameters=communicate_pb2.ClientGrpcMessage.SendParameters(
                    client_id=detail['client_id'],
                    train_dataset_length=detail['train_dataset_length'],
                    new_model_weight=serialize_model_state_dict(detail['new_model_weight']),
                    lora_config=lora_config_message
                )
            )
            response = stub.SendWeights(msg_proto)
        else:
            raise ValueError(f"Invalid message type: {message_type}")
        return response

    try:
        # Yield methods
        yield (receive, send)
    finally:
        # Make sure to have a final
        channel.close()
        logging.log(DEBUG, "gRPC channel closed")

