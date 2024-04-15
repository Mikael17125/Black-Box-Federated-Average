import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import FederatedSampler, SVHNDataset
from models import CustomCLIP
from utils import arg_parser
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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
        self.root_model.eval()
            
        self.criterion = nn.CrossEntropyLoss()        


    def test(self):
        
        target_layers = [self.root_model.image_encoder]
        cam = GradCAM(model=self.root_model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(9)]


        self.test_loader.sampler.set_client(0)
        import pdb;pdb.set_trace()

        for images, labels in self.test_loader:

            grayscale_cam = cam(input_tensor=images, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(images, grayscale_cam, use_rgb=True)

if __name__ == "__main__":
    args = arg_parser()
    fed_avg = FedAvg(args)
    fed_avg.test()
