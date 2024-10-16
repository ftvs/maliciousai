import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader, random_split

from torchvision.models.video import s3d, S3D_Weights
from torchvision.datasets.utils import download_url

import torchvision.transforms.v2 as transforms

import numpy as np

# custom libraries
from utils import *
from celebdf2 import *

# class MiniVGG(nn.Module):
#     def __init__(self, n_classes=10):
#         super(MiniVGG, self).__init__()

#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32, momentum=0.1),
#             nn.ReLU(inplace=True), # How to customize activation function?

#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32, momentum=0.1),
#             nn.ReLU(inplace=True),

#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(p=0.25),

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64, momentum=0.1),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64, momentum=0.1),
#             nn.ReLU(inplace=True),

#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(p=0.25),
#         )

#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 8 * 8, 512),
#             nn.BatchNorm1d(512, momentum=0.9),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(512, n_classes)
#         )

#         # How to customize weight initialization?

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x
    

# def train_vgg():
#     # Define a transform to normalize the data
#     transform = transforms.Compose(
#         [transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#     # Download and load the training set
#     train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                             download=True, transform=transform)
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
#                                             shuffle=True)

#     # Download and load the test set
#     test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                         download=True, transform=transform)
#     test_loader = torch.utils.data.DataLoader(test_set, batch_size=64,
#                                             shuffle=False)
#     # Split the training set into two halves
#     train_size = 5000
#     val_size = 5000
#     #split into three subsets train, valid, and the rest
#     train_subset, val_subset, _ = torch.utils.data.random_split(train_set, [train_size, val_size, len(train_set) - train_size - val_size])

#     # Create data loaders for the training and validation sets
#     train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64,
#                                             shuffle=True)
#     val_loader = torch.utils.data.DataLoader(val_subset, batch_size=64,
#                                                 shuffle=False)

#     model = MiniVGG()
#     model = model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     trainer = BaseTrainer(model, criterion, optimizer, train_loader, val_loader)
#     trainer.fit(num_epochs=1)

def train_s3d(dataset_path,batch_size,device,epochs):
    #%%
    model = s3d(weights=S3D_Weights.DEFAULT)
    freeze(model)
    # unfreeze(model)
    # replace final layer with new one with appropriate num of classes
    model.classifier[1] = nn.Conv3d(1024, 2, kernel_size=1, stride=1)

    # check if weights are forzen
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print(f"Parameter {name} is frozen.")
    #     else:
    #         print(f"Parameter {name} is trainable.")

    # class ConvertBCHWtoCBHW(nn.Module):
    #     """Convert tensor from (B, C, H, W) to (C, B, H, W)"""

    #     def forward(self, vid: torch.Tensor) -> torch.Tensor:
    #         return vid.permute(1, 0, 2, 3)

    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),    
        transforms.Normalize(mean=(0.43216, 0.394666, 0.37645),
                            std=(0.22803, 0.22145, 0.216989)),
        # transforms.CenterCrop(256),
        transforms.Resize((256,256))
        # ConvertBCHWtoCBHW()
    ])
    # transform = S3D_Weights.DEFAULT.transforms()

    # train_data = CelebDF2(dataset_path, transform=transform, max_frames=300, n_frames=150, file_list = 'List_of_training_videos.txt') # 10s @ 30fps = 300 frames, sample 15 frames per 1s (60,100,150)
    # val_data = CelebDF2(dataset_path, transform=transform, max_frames=300, n_frames=150, file_list = 'List_of_testing_videos.txt') # 10s @ 30fps = 300 frames, sample 15 frames per 1s (60,100,150)
    
    dataset = CelebDF2(dataset_path, transform=transform, max_frames=300, n_frames=150,file_list = 'List_of_testing_videos.txt') # 10s @ 30fps = 300 frames, sample 15 frames per 1s (60,100,150)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    print(f"Training size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size, pin_memory=True, num_workers=4, persistent_workers=True) #, num_workers=4, persistent_workers=True
    val_loader = DataLoader(val_data, batch_size, pin_memory=True, num_workers=4, persistent_workers=True) # , num_workers=4, persistent_workers=True # maybe just split from training

    first_data, first_labels = next(iter(train_loader))
    # print(first_data)
    print(f"Input Shape: {first_data.shape}")
    print(f"label Shape: {first_labels.shape}")

    # scale weights based on class distribution [(890-178),(5639-340)]
    # class_sample_counts = [712, 5299]  # Updated with your distribution
    class_sample_counts = np.array([178, 340])  # Updated with your distribution
    class_weights = sum(class_sample_counts) / torch.tensor((class_sample_counts*2), dtype=torch.float)

    trainer = BaseTrainer(
        model,
        nn.CrossEntropyLoss(weight=class_weights.to(device)),
        # nn.CrossEntropyLoss(),

        # SGD: overfit/train slow + good generalise well
        # optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=0.0005), #lr=1e-3,weight_decay=0.0005

        # Adam: overfits/train fast + generalise ok but may not be optimal
        # optim.Adam(model.parameters(), lr=1e-3,weight_decay=0.0005),

        # AdamW: overfits/train very fast + generalise ok
        optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01), #default: betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01/0.0005
        
        train_loader,
        val_loader,
        device = device)

    train_log, val_log = trainer.fit(epochs)
    # result = trainer.evaluate(val_loader)
    # print('test performance:', result)

    return train_log, val_log
    # return None,None