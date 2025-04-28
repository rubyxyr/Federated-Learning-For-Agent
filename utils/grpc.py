import grpc
from typing import Optional, Sequence
import logging
from logging import DEBUG

GRPC_MAX_MESSAGE_LENGTH: int = 536_870_912  # 512 * 1024 * 1024


def create_channel(
    server_address: str,
    insecure: bool,
    root_certificates: Optional[bytes] = None,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    interceptors: Optional[Sequence[grpc.UnaryUnaryClientInterceptor]] = None,
) -> grpc.Channel:
    """Create a gRPC channel, either secure or insecure."""
    # Check for conflicting parameters
    if insecure and root_certificates is not None:
        raise ValueError(
            "Invalid configuration: 'root_certificates' should not be provided "
            "when 'insecure' is set to True. For an insecure connection, omit "
            "'root_certificates', or set 'insecure' to False for a secure connection."
        )

    # Possible options:
    # https://github.com/grpc/grpc/blob/v1.43.x/include/grpc/impl/codegen/grpc_types.h
    channel_options = [
        ("grpc.max_send_message_length", max_message_length),
        ("grpc.max_receive_message_length", max_message_length),
    ]

    if insecure:
        channel = grpc.insecure_channel(server_address, options=channel_options)
        logging.log(DEBUG, "Opened insecure gRPC connection (no certificates were passed)")
    else:
        ssl_channel_credentials = grpc.ssl_channel_credentials(root_certificates)
        channel = grpc.secure_channel(
            server_address, ssl_channel_credentials, options=channel_options
        )
        logging.log(DEBUG, "Opened secure gRPC connection using certificates")

    if interceptors is not None:
        channel = grpc.intercept_channel(channel, interceptors)

    return channel
