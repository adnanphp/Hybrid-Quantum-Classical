"""Hybrid quantum-classical model (single class, config-driven)."""
import torch
import torch.nn as nn
import pennylane as qml
from pennylane.qnn import TorchLayer

from configs.config import DATASETS
from src.quantum.device import get_quantum_device
from src.quantum.encodings import CIRCUITS


class EnhancedHybridModel(nn.Module):
    def __init__(self, dataset_name, num_qubits=4, num_layers=2, encoding='amplitude'):
        super().__init__()
        cfg = DATASETS[dataset_name]
        self.input_channels = cfg['input_channels']
        self.num_classes = cfg['num_classes']
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.encoding = encoding

        # -- classical trunk (identical for all variants except output width) --
        if dataset_name == 'MNIST':
            self.classical_net = nn.Sequential(
                nn.Conv2d(self.input_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),                  nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            pre_q_in = 32 * 4 * 4
        elif dataset_name == 'CIFAR100':
            self.classical_net = nn.Sequential(
                nn.Conv2d(self.input_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),                  nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),                 nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            pre_q_in = 128 * 4 * 4
        else:  # STL10
            self.classical_net = nn.Sequential(
                nn.Conv2d(self.input_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),                  nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),                 nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),                nn.BatchNorm2d(256), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            pre_q_in = 256 * 4 * 4

        # -- feature reducer: size & activation depend on encoding --
        if encoding == 'amplitude':
            q_in = 2 ** num_qubits                     # 16 for 4q
            mid = 256 if dataset_name == 'MNIST' else (512 if dataset_name == 'CIFAR100' else 1024)
            self.feature_reducer = nn.Sequential(
                nn.Linear(pre_q_in, mid), nn.ReLU(),
                nn.Linear(mid, q_in), nn.Softmax(dim=1),
            )
        else:  # angle
            q_in = num_qubits                          # 2 for 2q
            mid = 128 if dataset_name == 'MNIST' else 256
            self.feature_reducer = nn.Sequential(
                nn.Linear(pre_q_in, mid), nn.ReLU(),
                nn.Linear(mid, q_in), nn.Tanh(),
            )

        # -- quantum layer --
        dev = get_quantum_device(num_qubits)
        circuit = CIRCUITS[encoding]
        qnode = qml.QNode(circuit, dev, interface="torch")
        self.qlayer = TorchLayer(qnode, {"weights": (num_layers, num_qubits)})

        # -- head --
        self.head = nn.Sequential(
            nn.Linear(num_qubits, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, self.num_classes),
        )

    def forward(self, x):
        x = self.classical_net(x)
        x = x.view(x.size(0), -1)
        x = self.feature_reducer(x)
        q = self.qlayer(x)
        return self.head(q)
