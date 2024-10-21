import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader, random_split

from torchvision.models.video import s3d, S3D_Weights
from torchvision.datasets.utils import download_url

import torchvision.transforms.v2 as transforms

import numpy as np
import random
import copy

# custom libraries
from utils import *
from celebdf2 import *


def train_s3d(dataset_path,batch_size,device,epochs):
    # Set the random seed for reproducibility
    seed = 50  # You can choose any integer for your seed value

    # Set the seed for PyTorch (CPU and CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setup

    # Set the seed for Python's random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Initialize the S3D model
    model = s3d(weights=S3D_Weights.DEFAULT)
    # freeze(model)
    unfreeze(model)
    # replace final layer with new one with appropriate num of classes
    model.classifier[1] = nn.Conv3d(1024, 2, kernel_size=1, stride=1)

    # check if weights are forzen
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print(f"Parameter {name} is frozen.")
    #     else:
    #         print(f"Parameter {name} is trainable.")

    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),    
        transforms.Normalize(mean=(0.43216, 0.394666, 0.37645),
                            std=(0.22803, 0.22145, 0.216989)),
        # transforms.CenterCrop(256),
        transforms.Resize((256,256))
    ])
    # transform = S3D_Weights.DEFAULT.transforms()

    # train_data = CelebDF2(dataset_path, transform=transform, max_frames=300, n_frames=150, file_list = 'List_of_training_videos.txt') # 10s @ 30fps = 300 frames, sample 15 frames per 1s (60,100,150)
    # val_data = CelebDF2(dataset_path, transform=transform, max_frames=300, n_frames=150, file_list = 'List_of_testing_videos.txt') # 10s @ 30fps = 300 frames, sample 15 frames per 1s (60,100,150)
    
    dataset = CelebDF2(dataset_path, transform=transform, max_frames=300, n_frames=150,file_list = 'List_of_testing_videos.txt') # 10s @ 30fps = 300 frames, sample 15 frames per 1s (60,100,150)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    print(f"Training size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size, pin_memory=True, num_workers=4, persistent_workers=True, shuffle=True) #, num_workers=4, persistent_workers=True
    val_loader = DataLoader(val_data, batch_size, pin_memory=True, num_workers=4, persistent_workers=True, shuffle=False) # , num_workers=4, persistent_workers=True # maybe just split from training

    first_data, first_labels = next(iter(train_loader))
    # print(first_data)
    print(f"Input Shape: {first_data.shape}")
    print(f"label Shape: {first_labels.shape}")

    # scale weights based on class distribution [(5639-340),(890-178)]
    # class_sample_counts = np.array([5299, 712])  # Updated with your distribution
    class_sample_counts = np.array([340, 178])  # Updated with your distribution
    class_weights = 1 / torch.tensor((class_sample_counts), dtype=torch.float)
    # class_weights = sum(class_sample_counts) / torch.tensor((class_sample_counts), dtype=torch.float)
    class_weights = sum(class_sample_counts) / torch.tensor((class_sample_counts*2), dtype=torch.float)
    # class_weights[1] = class_weights[1]*2
    
    
    trainer = BaseTrainer(
        model,
        nn.CrossEntropyLoss(weight=class_weights.to(device)),

        # SGD: overfit/train slow + good generalise well
        optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=0.0005), #lr=1e-3,weight_decay=0.0005

        # Adam: overfits/train fast + generalise ok but may not be optimal
        # optim.Adam(model.parameters(), lr=1e-3), #,weight_decay=0.0005

        # AdamW: overfits/train very fast + generalise ok
        # optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01), #default: betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01/0.0005
        
        train_loader,
        val_loader,
        device = device,
        validation_interval=300)

    train_log, val_log = trainer.fit(epochs)
    # result = trainer.evaluate(val_loader)
    # print('test performance:', result)

    return train_log, val_log
    # return None,None

def eval_s3d(dataset_path,batch_size,device,model_path):
    # Set the random seed for reproducibility
    seed = 50  # You can choose any integer for your seed value

    # Set the seed for PyTorch (CPU and CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setup

    # Set the seed for Python's random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),    
        transforms.Normalize(mean=(0.43216, 0.394666, 0.37645),
                            std=(0.22803, 0.22145, 0.216989)),
        # transforms.CenterCrop(256),
        transforms.Resize((256,256))
    ])
    # transform = S3D_Weights.DEFAULT.transforms()

    # train_data = CelebDF2(dataset_path, transform=transform, max_frames=300, n_frames=150, file_list = 'List_of_training_videos.txt') # 10s @ 30fps = 300 frames, sample 15 frames per 1s (60,100,150)
    # val_data = CelebDF2(dataset_path, transform=transform, max_frames=300, n_frames=150, file_list = 'List_of_testing_videos.txt') # 10s @ 30fps = 300 frames, sample 15 frames per 1s (60,100,150)
    
    dataset = CelebDF2(dataset_path, transform=transform, max_frames=300, n_frames=150,file_list = 'List_of_testing_videos.txt') # 10s @ 30fps = 300 frames, sample 15 frames per 1s (60,100,150)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    print(f"Training size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size, pin_memory=True, num_workers=4, persistent_workers=True, shuffle=False) #, num_workers=4, persistent_workers=True
    val_loader = DataLoader(val_data, batch_size, pin_memory=True, num_workers=4, persistent_workers=True, shuffle=False) # , num_workers=4, persistent_workers=True # maybe just split from training

    first_data, first_labels = next(iter(train_loader))
    # print(first_data)
    print(f"Input Shape: {first_data.shape}")
    print(f"label Shape: {first_labels.shape}")

    # scale weights based on class distribution [(5639-340),(890-178)]
    # class_sample_counts = np.array([5299, 712])  # Updated with your distribution
    class_sample_counts = np.array([340, 178])  # Updated with your distribution
    # class_weights = 1 / torch.tensor((class_sample_counts), dtype=torch.float)
    class_weights = sum(class_sample_counts) / torch.tensor((class_sample_counts*2), dtype=torch.float)

    # load saved model
    checkpoint = torch.load(model_path, weights_only=False)

    # Initialize the S3D model & replace final layer with new one with appropriate num of classes
    # model = s3d(weights=S3D_Weights.DEFAULT)
    # model.classifier[1] = nn.Conv3d(1024, 2, kernel_size=1, stride=1)

    # saved_weights = copy.deepcopy(model.state_dict())
    # saved_weights = copy.deepcopy(checkpoint['model_state_dict'])

    # model.load_state_dict(checkpoint['model_state_dict'])
    model = checkpoint['model']
    freeze(model)

    # Compare the keys and values
    # for key in saved_weights:
    #     if torch.equal(saved_weights[key], model.state_dict()[key]):
    #         print(f"Weights for layer {key} match.")
        
    #     else:
    #         print(f"Weights for layer {key} do not match.")
            

    # optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3), #,weight_decay=0.0005
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=0.0005) #lr=1e-3,weight_decay=0.0005
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    evaluater = BaseTrainer(
        model,
        nn.CrossEntropyLoss(weight=class_weights.to(device)),
        optimizer, 
        train_loader,
        val_loader,
        device = device,
        validation_interval=100)
    
    train_loss, train_accuracy, train_cm = evaluater.validate_train_loader()
    val_loss, val_accuracy, val_cm = evaluater.validate_one_epoch()

    return train_loss, train_accuracy, train_cm ,val_loss, val_accuracy, val_cm