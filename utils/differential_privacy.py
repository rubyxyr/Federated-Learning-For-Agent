import numpy as np
import torch


def compute_stdv(
    noise_multiplier: float, clipping_norm: float, num_sampled_clients: int
) -> float:
    """Compute standard deviation for noise addition.

    Paper: https://arxiv.org/abs/1710.06963
    """
    return float((noise_multiplier * clipping_norm) / num_sampled_clients)


def add_gaussian_noise(input_data, noise_multiplier, clipping_norm, num_sampled_clients, device=None):
    return local_add_gaussian_noise(input_data,
                                    compute_stdv(
                                        noise_multiplier,
                                        clipping_norm,
                                        num_sampled_clients
                                    ),
                                    device
                                    )


def local_add_gaussian_noise(input_data, std_dev, device):
    for k, layer in input_data.items():
        input_data[k] = layer.to(device) + torch.normal(0, std_dev, layer.shape).to(device)
    return input_data


def get_norm(input_array):
    """Calculates the L2-norm of a potentially ragged array"""
    flattened_update = input_array[0]
    for i in range(1, len(input_array)):
        flattened_update = np.append(flattened_update, input_array[i])
    return float(np.sqrt(np.sum(np.square(flattened_update))))


def clip_l2_norm(client_parameter, server_parameter, clip_threshold, device):
    """Scales the update so its L2 norm is upper-bound to threshold."""
    c_p = [tensor.numpy(force=True) for tensor in list(client_parameter.values())]
    s_p = [tensor.numpy(force=True) for tensor in list(server_parameter.values())]
    update = [np.subtract(x, y) for (x, y) in zip(c_p, s_p)]
    update_norm = get_norm(update)

    scaling_factor = min(1, clip_threshold / update_norm)
    update_clipped = [torch.from_numpy(layer * scaling_factor).to(device) for layer in update]
    for i, k in enumerate(server_parameter.keys()):
        client_parameter[k] = server_parameter[k] + update_clipped[i]
    return client_parameter, (scaling_factor < 1)
