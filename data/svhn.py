from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

class SVHNDataset:
    def __init__(self):
                
        self.train_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
                                # transforms.RandomResizedCrop((224, 224), (0.08, 1.0), interpolation=InterpolationMode.BICUBIC, antialias=True),
                                transforms.RandomHorizontalFlip(),
                                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                                     (0.26862954, 0.26130258, 0.27577711))])
        
        self.test_transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Resize(max(224,224), interpolation=InterpolationMode.BICUBIC, antialias=True),
                        transforms.CenterCrop(max(224,224)), 
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                                (0.26862954, 0.26130258, 0.27577711))])

    def get_train_data(self):
        full_train_dataset = datasets.SVHN(root='dataset', 
                                           split='train', 
                                           transform=self.train_transform, 
                                           download=True)

        return full_train_dataset
        

    def get_test_data(self):
        full_test_dataset = datasets.SVHN(root='dataset', 
                                          split='test', 
                                          transform=self.test_transform,
                                          download=True)
        
        
        return full_test_dataset