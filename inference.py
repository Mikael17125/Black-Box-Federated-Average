import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import FederatedSampler, SVHNDataset
from models import CustomCLIP
from utils import arg_parser
import matplotlib.pyplot as plt
from tqdm import tqdm

class FedAvg:

    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        self.dataset = SVHNDataset()

        self.test_set = self.dataset.get_test_data()

        sampler = FederatedSampler(self.test_set, 
                                    non_iid=self.args.non_iid, 
                                    n_clients=self.args.n_clients, 
                                    n_shards=self.args.n_shards,
                                    sampler_path="sampler/test_few_shot_iid_seed_zero.json")

        self.test_loader = DataLoader(self.test_set, 
                                        batch_size=16, 
                                        sampler=sampler)
        
        checkpoint = torch.load("results/analyze_few_0.1/model_state.pth")
        self.root_model = CustomCLIP().to(self.device)
        self.root_model.coordinator.dec.load_state_dict(checkpoint)
        self.target_acc = 0.99
            
        self.criterion = nn.CrossEntropyLoss()


    def _test_clients(self, root_model, test_loader):

        correct = 0
        len_data = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = root_model(images)

                correct += (labels == torch.argmax(outputs, dim=1)).float().sum()
                len_data += outputs.shape[0]

        return correct/len_data
        


    def test(self):

        idx_clients = np.arange(0,100,1)
        clients_acc = {}

        for client_idx in tqdm(idx_clients):
            self.test_loader.sampler.set_client(client_idx)
            clients_acc[client_idx] = self._test_clients(self.root_model, self.test_loader).item()

        client_indices = list(clients_acc.keys())
        client_accuracies = list(clients_acc.values())
        average_accuracy = np.mean(client_accuracies)

        colors = ['red' if acc < average_accuracy else 'blue' for acc in client_accuracies]

        plt.bar(client_indices, client_accuracies, color=colors)
        plt.axhline(y=average_accuracy, color='green', linestyle='--', label=f'Mean: {average_accuracy:.2f}')
        plt.xlabel('Client Index')
        plt.ylabel('Accuracy')
        plt.title('Client Accuracy')
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    args = arg_parser()
    fed_avg = FedAvg(args)
    fed_avg.test()
