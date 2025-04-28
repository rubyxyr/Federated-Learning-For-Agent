import os
import torch
from omegaconf import OmegaConf
from .strategy import Strategy
from utils.differential_privacy import add_gaussian_noise, clip_l2_norm


class DpServerFixedClip(Strategy):
    def __init__(self, cfg_path: str, strategy: Strategy):
        super().__init__()
        self.config_detail = OmegaConf.load(cfg_path)
        self.strategy = strategy
        self.noise_multiplier = self.config_detail.server.noise_multiplier
        self.clip_threshold = self.config_detail.server.clip_threshold
        self.params_current = None
        self.device_map = self.config_detail.model.device_map

    def set_model_parameters(self, parameter):
        self.params_current = parameter

    def aggregate(self, client_list, dataset_len_list, weight_path_list, clients_weights_dict=None):
        clients_weights_dict = dict()
        for k, p in enumerate(weight_path_list):
            single_output_dir = os.path.join(
                p,
                "pytorch_model.bin",
            )
            single_weights = torch.load(single_output_dir, map_location=torch.device(self.device_map))
            single_weights, _ = clip_l2_norm(single_weights,
                                             self.params_current,
                                             self.clip_threshold,
                                             self.device_map)
            clients_weights_dict[p] = single_weights
        new_weight = self.strategy.aggregate(client_list=client_list,
                                             dataset_len_list=dataset_len_list,
                                             weight_path_list=weight_path_list,
                                             clients_weights_dict=clients_weights_dict)
        if new_weight is not None:
            new_weight = add_gaussian_noise(new_weight,
                                            self.noise_multiplier,
                                            self.clip_threshold,
                                            len(client_list),
                                            device=self.device_map)
        return new_weight
