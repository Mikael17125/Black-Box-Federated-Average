from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import Sampler
import json

class FederatedSampler(Sampler):
    def __init__(
        self,
        dataset: Sequence,
        non_iid: int,
        sampler_path: str = None,
        n_clients: Optional[int] = 100,
        n_shards: Optional[int] = 200
    ):
        """Sampler for federated learning in both iid and non-iid settings.

        Args:
            dataset (Sequence): Dataset to sample from.
            non_iid (int): 0: IID, 1: Non-IID
            n_clients (Optional[int], optional): Number of clients. Defaults to 100.
            n_shards (Optional[int], optional): Number of shards. Defaults to 200.
        """
        self.dataset = dataset
        self.non_iid = non_iid
        self.n_clients = n_clients
        self.n_shards = n_shards
        self.sampler_path = sampler_path

        if self.non_iid:
            self.dict_users = self._sample_non_iid()
        else:
            if self.sampler_path != None:
                self.dict_users = self._load_sample_iid()
            else:
                self.dict_users = self._create_sample_iid()

    def _load_sample_iid(self):
        print("LOAD SAMPLER ")
        with open(self.sampler_path, 'r') as json_file:
            dict_users_loaded = json.load(json_file)

        # Convert lists back to sets if needed
        dict_users = {int(key): value for key, value in dict_users_loaded.items()}
        return dict_users

    def _create_sample_iid(self) -> Dict[int, List[int]]:
        print("CREATE SAMPLER")
        num_shots = 16
        dict_users, all_idxs = {}, [i for i in range(len(self.dataset))]

        for i in range(self.n_clients):
            dict_users[i] = set(np.random.choice(all_idxs, num_shots, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])

        dict_users_serializable = {k: [int(idx) for idx in v] for k, v in dict_users.items()}

        json_file_path = 'sampler/new_sampler.json'
        with open(json_file_path, 'w') as json_file:
            json.dump(dict_users_serializable, json_file)

        return dict_users

    def _sample_non_iid(self) -> Dict[int, List[int]]:
        num_imgs = len(self.dataset) // self.n_shards  # 300

        idx_shard = [i for i in range(self.n_shards)]
        dict_users = {i: np.array([]) for i in range(self.n_clients)}
        idxs = np.arange(self.n_shards * num_imgs)
        labels = self.dataset.train_labels.numpy()

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]
        

        # divide and assign 2 shards/client
        for i in range(self.n_clients):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )

        return dict_users

    def set_client(self, client_id: int):
        self.client_id = client_id

    def __iter__(self):
        # fetch dataset indexes based on current client
        client_idxs = list(self.dict_users[self.client_id])
        for item in client_idxs:
            yield int(item)
