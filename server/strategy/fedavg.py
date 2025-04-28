from server.strategy.strategy import Strategy
from utils.model_agg import fed_average


class FedAvg(Strategy):
    def __init__(self):
        super().__init__()
        self.params_current = None

    def aggregate(self, client_list, dataset_len_list, weight_path_list, clients_weights_dict=None):
        return fed_average(dataset_len_list, weight_path_list, clients_weights_dict)

    def set_model_parameters(self, parameter):
        self.params_current = parameter
