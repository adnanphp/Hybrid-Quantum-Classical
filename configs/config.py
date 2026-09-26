"""
Central configuration.

Every experiment (dataset + encoding + qubit count) is described by an
EXPERIMENTS entry.  Adding a new variant = adding a new dict entry here.
No other file needs to change.
"""
import torch
from torchvision import transforms, datasets

# ---------------------------------------------------------------------------
# Global
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Dataset registry (transforms / channels / classes)
# ---------------------------------------------------------------------------
DATASETS = {
    'MNIST': {
        'loader': datasets.MNIST,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
        ]),
        'input_channels': 1,
        'num_classes': 10,
        'class_names': [str(i) for i in range(10)],
    },
    'CIFAR100': {
        'loader': datasets.CIFAR100,
        'transform': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ]),
        'input_channels': 3,
        'num_classes': 100,
        'class_names': [str(i) for i in range(100)],
    },
    'STL10': {
        'loader': datasets.STL10,
        'transform': transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(96, padding=8),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066),
                                 (0.2603, 0.2566, 0.2713)),
        ]),
        'input_channels': 3,
        'num_classes': 10,
        'class_names': ['airplane', 'bird', 'car', 'cat', 'deer',
                        'dog', 'horse', 'monkey', 'ship', 'truck'],
    },
}

# ---------------------------------------------------------------------------
# Experiment registry
#
#   name            : folder name under results/
#   dataset         : key in DATASETS
#   num_qubits      : wires in the quantum circuit
#   num_layers      : variational layers
#   encoding        : 'amplitude' | 'angle'
#   epochs          : training epochs
#   batch_size      : minibatch size
#   lr_hybrid / lr_classical / max_lr_hybrid
#   plot_format     : 'eps' | 'png'
#   early_stop      : (min_hybrid_acc, min_classical_acc)
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    'mnist_4q': {
        'dataset': 'MNIST',
        'num_qubits': 4,
        'num_layers': 2,
        'encoding': 'amplitude',
        'epochs': 50,
        'batch_size': 64,
        'lr_hybrid': 0.001,
        'lr_classical': 0.003,
        'max_lr_hybrid': 0.002,
        'plot_format': 'eps',
        'early_stop': (80, 90),
    },
    'cifar100_4q': {
        'dataset': 'CIFAR100',
        'num_qubits': 4,
        'num_layers': 2,
        'encoding': 'amplitude',
        'epochs': 50,
        'batch_size': 64,
        'lr_hybrid': 0.001,
        'lr_classical': 0.003,
        'max_lr_hybrid': 0.002,
        'plot_format': 'eps',
        'early_stop': (5, 15),
    },
    'stl10_4q': {
        'dataset': 'STL10',
        'num_qubits': 4,
        'num_layers': 2,
        'encoding': 'amplitude',
        'epochs': 50,
        'batch_size': 64,
        'lr_hybrid': 0.001,
        'lr_classical': 0.003,
        'max_lr_hybrid': 0.002,
        'plot_format': 'eps',
        'early_stop': (30, 40),
    },
    'mnist_2q_angle': {
        'dataset': 'MNIST',
        'num_qubits': 2,
        'num_layers': 2,
        'encoding': 'angle',
        'epochs': 10,
        'batch_size': 64,
        'lr_hybrid': 0.001,
        'lr_classical': 0.003,
        'max_lr_hybrid': 0.002,
        'plot_format': 'png',
        'early_stop': (80, 90),
    },
}

# Backwards-compat convenience constants
DEFAULT_EXPERIMENT = 'mnist_4q'
