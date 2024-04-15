import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import FederatedSampler, SVHNDataset
from models import CustomCLIP
from utils import arg_parser, average_weights
from spsa import SPSA
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd


class FedAvg:

    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        self.dataset = SVHNDataset()
        self.train_set = self.dataset.get_train_data()
        self.test_set = self.dataset.get_test_data()
        self.writer = SummaryWriter(log_dir=args.log_dir)

        sampler = FederatedSampler(self.train_set, 
                                    non_iid=self.args.non_iid, 
                                    n_clients=self.args.n_clients, 
                                    n_shards=self.args.n_shards,
                                    sampler_path="sampler/train_few_shot_iid_seed_zero.json")

        self.train_loader = DataLoader(self.train_set, 
                                        batch_size=128, 
                                        sampler=sampler)

        self.test_loader = DataLoader(self.test_set, 
                                        batch_size=128)

        self.root_model = CustomCLIP().to(self.device)
        self.target_acc = 0.99
            
        self.criterion = nn.CrossEntropyLoss()


    def _train_client(self, root_model, train_loader):
        model = copy.deepcopy(root_model)
        spsa = SPSA(model, self.criterion)

        for epoch in (range(1, self.args.n_client_epochs+1)):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            for idx, (images, labels) in enumerate((train_loader)):

                images = images.to(self.device)
                labels = labels.to(self.device)
                
                spsa.estimate(epoch, images, labels)
                outputs = spsa.model(images)
                loss = spsa.criterion(outputs, labels)
                
                epoch_loss += loss.item()
                epoch_correct += (outputs.argmax(dim=1) == labels).sum().item()
                epoch_samples += outputs.shape[0]

            epoch_loss /= idx+1
            epoch_acc = epoch_correct / epoch_samples

            print(f"\t\tEpoch: {epoch}/{self.args.n_client_epochs} | Loss: {epoch_loss} | Acc: {epoch_acc}")
        
        return spsa.model.coordinator.dec, epoch_acc


    def train(self):
        clients_acc = {}
        for i in range(self.args.n_clients):
            clients_acc[i] = []

        for epoch in range(self.args.n_commmunication_rounds):
            print(f"Round: {epoch+1}/{self.args.n_commmunication_rounds}")
            
            idx = 0
            clients_decoders = []

            for i in range(self.args.n_clients):
                clients_acc[i].append('n')

            m = max(int(self.args.frac * self.args.n_clients), 1)
            idx_clients = np.random.choice(range(self.args.n_clients), m, replace=False)

            for client_idx in idx_clients:
                print(f"\tClient: {client_idx} ({idx+1}/{len(idx_clients)})")
                
                self.train_loader.sampler.set_client(client_idx)
                client_decoder, client_acc = self._train_client(self.root_model, self.train_loader)
                clients_decoders.append(client_decoder.state_dict())
                clients_acc[client_idx][epoch] = client_acc

                idx += 1

            updated_weights = average_weights(clients_decoders)
            self.root_model.coordinator.dec.load_state_dict(updated_weights) 

            if (epoch + 1) % self.args.log_every == 0:
                total_loss, total_acc = self.server_test()
                if total_acc >= self.target_acc:
                    print(f"\n -----> Target accuracy {self.target_acc} reached at round {epoch}! <----- \n")

                print(f"\n\nResults after {epoch + 1} rounds of training:")
                print(f"---> Avg Test Loss: {total_loss} | Avg Test Accuracy: {total_acc}\n")

                self.writer.add_scalar('Test/Loss', total_loss, global_step=epoch + 1)
                self.writer.add_scalar('Test/Accuracy', total_acc, global_step=epoch + 1)

                torch.save(self.root_model.coordinator.dec.state_dict(), os.path.join(self.args.save_dir, "model_state.pth"))

                clients_acc_df = pd.DataFrame(clients_acc.items(), columns=['client_id', 'accuracy'])
                # Save the DataFrame to a CSV file
                output_path = os.path.join(self.args.save_dir, "clients_outputs.csv")
                clients_acc_df.to_csv(output_path, index=False)

        self.writer.close()

    def server_test(self):

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        spsa = SPSA(self.root_model, self.criterion)
        with torch.no_grad():
            for idx, (images, labels) in enumerate(tqdm(self.test_loader)):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = spsa.model(images)
                total_loss += spsa.criterion(outputs, labels).item()

                total_correct += (labels == torch.argmax(outputs, dim=1)).float().sum()
                total_samples += outputs.shape[0]

            total_loss /= idx
            total_acc = total_correct / total_samples

        return total_loss, total_acc

if __name__ == "__main__":
    args = arg_parser()
    fed_avg = FedAvg(args)
    fed_avg.train()
