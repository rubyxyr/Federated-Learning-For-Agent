import torch
import os
from torch.nn.functional import normalize


# def fed_average(
#     selected_clients_set, output_dir, local_dataset_len_dict, version, clients_params_dict=None
# ):
#     r"""
#     selected_clients_set:选中客户端的集合列表
#     output_dir: 权重输出文件夹
#     local_dataset_len_dict: 每个客户端的数据集大小
#     version: 当前版本
#     clients_params_dict: 多个客户端权重集合
#     """
#     weights_array = normalize(
#         torch.tensor(
#             [local_dataset_len_dict[client_id] for client_id in selected_clients_set],
#             dtype=torch.float32,
#         ),
#         p=1,
#         dim=0,
#     )
#     for k, client_id in enumerate(selected_clients_set):
#         single_output_dir = os.path.join(
#             output_dir,
#             str(version),
#             "local_output_{}".format(client_id),
#             "pytorch_model.bin",
#         )
#         single_weights = torch.load(single_output_dir) if clients_params_dict is None else clients_params_dict[client_id]
#         if k == 0:
#             weighted_single_weights = {
#                 key: single_weights[key] * (weights_array[k])
#                 for key in single_weights.keys()
#             }
#         else:
#             weighted_single_weights = {
#                 key: weighted_single_weights[key]
#                 + single_weights[key] * (weights_array[k])
#                 for key in single_weights.keys()
#             }
#
#     # set_peft_model_state_dict(model, weighted_single_weights, "default")
#     return weighted_single_weights


def fed_average(dataset_len_list, file_path_list, clients_weights_dict=None):
    weights_array = normalize(
        torch.tensor(
            dataset_len_list,
            dtype=torch.float32,
        ),
        p=1,
        dim=0,
    )
    for k, p in enumerate(file_path_list):
        single_output_dir = os.path.join(
            p,
            "pytorch_model.bin",
        )
        single_weights = torch.load(single_output_dir) if clients_weights_dict is None else clients_weights_dict[p]
        if k == 0:
            weighted_single_weights = {
                key: single_weights[key] * (weights_array[k])
                for key in single_weights.keys()
            }
        else:
            weighted_single_weights = {
                key: weighted_single_weights[key]
                + single_weights[key] * (weights_array[k])
                for key in single_weights.keys()
            }

    return weighted_single_weights
