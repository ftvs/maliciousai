import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# custom library
from utils import BaseTrainer

class MiniVGG(nn.Module):
    def __init__(self, n_classes=10):
        super(MiniVGG, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.ReLU(inplace=True), # How to customize activation function?

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.BatchNorm1d(512, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, n_classes)
        )

        # How to customize weight initialization?

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a transform to normalize the data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Download and load the training set
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
                                          shuffle=True)

# Download and load the test set
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64,
                                         shuffle=False)
# Split the training set into two halves
train_size = 5000
val_size = 5000
#split into three subsets train, valid, and the rest
train_subset, val_subset, _ = torch.utils.data.random_split(train_set, [train_size, val_size, len(train_set) - train_size - val_size])

# Create data loaders for the training and validation sets
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64,
                                          shuffle=True)
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=64,
                                               shuffle=False)

model = MiniVGG()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trainer = BaseTrainer(model, criterion, optimizer, train_loader, val_loader)
trainer.fit(num_epochs=1)