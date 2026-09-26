"""Classical baseline CNN."""
import torch.nn as nn
from configs.config import DATASETS


class EnhancedClassicalCNN(nn.Module):
    def __init__(self, dataset_name):
        super().__init__()
        cfg = DATASETS[dataset_name]
        self.input_channels = cfg['input_channels']
        self.num_classes = cfg['num_classes']

        if dataset_name == 'MNIST':
            self.conv_net = nn.Sequential(
                nn.Conv2d(self.input_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),                  nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.head = nn.Sequential(
                nn.Linear(64 * 4 * 4, 128), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(128, self.num_classes),
            )
        elif dataset_name == 'CIFAR100':
            self.conv_net = nn.Sequential(
                nn.Conv2d(self.input_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),                 nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),                nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(256, 512, 3, padding=1),                nn.BatchNorm2d(512), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Sequential(
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(256, self.num_classes),
            )
        else:  # STL10
            self.conv_net = nn.Sequential(
                nn.Conv2d(self.input_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),                 nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),                nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(256, 512, 3, padding=1),                nn.BatchNorm2d(512), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Sequential(
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
                nn.Dropout(0.4), nn.Linear(256, self.num_classes),
            )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        return self.head(x)
