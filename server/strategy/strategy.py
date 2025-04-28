from abc import ABC, abstractmethod


class Strategy(ABC):

    @abstractmethod
    def aggregate(self, client_list, dataset_len_list, weight_path_list, clients_weights_dict=None):
        """Aggregate results from clients

        :param client_list client id list
        :param dataset_len_list: Updated dataset length list
        :param weight_path_list: weight file path list
        :param clients_weights_dict: clients weights dict
        :return: new model parameters
        """
        pass
