import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pennylane as qml
from pennylane.qnn import TorchLayer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import time
import psutil
import GPUtil
import copy
from sklearn.metrics import precision_score, recall_score, f1_score
from torchattacks import PGD
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_qubits = 4  # Changed from 2 to 4 qubits (2^4=16 amplitudes needed)
num_layers = 2
batch_size = 64
epochs = 50

# Loss function definition
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Dataset Configuration
dataset_config = {
    'MNIST': {
        'loader': datasets.MNIST,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1))
        ]),
        'input_channels': 1,
        'num_classes': 10
    },
    'CIFAR100': {
        'loader': datasets.CIFAR100,
        'transform': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]),
        'input_channels': 3,
        'num_classes': 100
    },
    'STL10': {
        'loader': datasets.STL10,
        'transform': transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(96, padding=8),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
        ]),
        'input_channels': 3,
        'num_classes': 10
    }
}

# Visualization Class with all requested modifications
class MLVisualizer:
    def __init__(self):
        self.metrics_history = {
            'Hybrid': {'train_loss': [], 'val_loss': [], 'accuracy': [], 
                      'precision': [], 'recall': [], 'f1': [], 'robustness': [],
                      'time': [], 'cpu': [], 'memory': []},
            'Classical': {'train_loss': [], 'val_loss': [], 'accuracy': [], 
                         'precision': [], 'recall': [], 'f1': [], 'robustness': [],
                         'time': [], 'cpu': [], 'memory': []},
            'Hybrid_Test': {'val_loss': [], 'accuracy': [], 
                           'precision': [], 'recall': [], 'f1': []},
            'Classical_Test': {'val_loss': [], 'accuracy': [], 
                              'precision': [], 'recall': [], 'f1': []}
        }
    
    def update_metrics(self, model_type, metrics_dict):
        if model_type not in self.metrics_history:
            print(f"Warning: Model type '{model_type}' not found in metrics history")
            return
            
        for key, value in metrics_dict.items():
            if key in self.metrics_history[model_type]:
                self.metrics_history[model_type][key].append(value)
            else:
                print(f"Warning: Metric '{key}' not found for model type '{model_type}'")
    
    def plot_training_curves(self, base_filename='training_curves'):
        """Generate separate plots for each metric type"""
        # Plot loss curves - now separate plots
        plt.figure(figsize=(8, 6))
        plt.plot(self.metrics_history['Hybrid']['train_loss'], 'b-', label='Hybrid Train')
        plt.plot(self.metrics_history['Hybrid']['val_loss'], 'b--', label='Hybrid Val')
        plt.plot(self.metrics_history['Classical']['train_loss'], 'r-', label='Classical Train')
        plt.plot(self.metrics_history['Classical']['val_loss'], 'r--', label='Classical Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training/Validation Loss Comparison')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{base_filename}_loss.eps', format='eps', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Plot accuracy - separate plot
        plt.figure(figsize=(8, 6))
        plt.plot(self.metrics_history['Hybrid']['accuracy'], 'g-', label='Hybrid')
        plt.plot(self.metrics_history['Classical']['accuracy'], 'm-', label='Classical')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Accuracy Comparison')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{base_filename}_accuracy.eps', format='eps', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Plot F1 score - separate plot
        plt.figure(figsize=(8, 6))
        plt.plot(self.metrics_history['Hybrid']['f1'], 'c-', label='Hybrid')
        plt.plot(self.metrics_history['Classical']['f1'], 'y-', label='Classical')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Comparison')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{base_filename}_f1.eps', format='eps', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Plot robustness if available - separate plot
        if any(self.metrics_history['Hybrid']['robustness']):
            plt.figure(figsize=(8, 6))
            xvals = range(5, len(self.metrics_history['Hybrid']['robustness'])*5 +1, 5)
            plt.plot(xvals, self.metrics_history['Hybrid']['robustness'], 'k-', label='Hybrid')
            plt.plot(xvals, self.metrics_history['Classical']['robustness'], 'k--', label='Classical')
            plt.xlabel('Epoch')
            plt.ylabel('Robustness (%)')
            plt.title('Adversarial Robustness Comparison')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{base_filename}_robustness.eps', format='eps', bbox_inches='tight', dpi=300)
            plt.close()

    def plot_resource_usage(self):
        """Separate plots for each resource metric"""
        # Time comparison
        plt.figure(figsize=(8, 6))
        plt.plot(self.metrics_history['Hybrid']['time'], 'b-', label='Hybrid')
        plt.plot(self.metrics_history['Classical']['time'], 'r-', label='Classical')
        plt.title('Training Time per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Time (s)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('resource_time.eps', format='eps', bbox_inches='tight', dpi=300)
        plt.close()
        
        # CPU usage
        plt.figure(figsize=(8, 6))
        plt.plot(self.metrics_history['Hybrid']['cpu'], 'b-', label='Hybrid')
        plt.plot(self.metrics_history['Classical']['cpu'], 'r-', label='Classical')
        plt.title('CPU Usage per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('CPU Usage (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('resource_cpu.eps', format='eps', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Memory usage
        plt.figure(figsize=(8, 6))
        plt.plot(self.metrics_history['Hybrid']['memory'], 'b-', label='Hybrid')
        plt.plot(self.metrics_history['Classical']['memory'], 'r-', label='Classical')
        plt.title('Memory Usage per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Memory (GB)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('resource_memory.eps', format='eps', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_confusion_matrix(self, model, data_loader, device, class_names, model_name):
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'confusion_matrix_{model_name}.eps', format='eps', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_class_distribution(self, dataset, title):
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
        else:
            targets = [y for _, y in dataset]
        
        plt.figure(figsize=(12, 8))
        sns.countplot(x=targets)
        plt.title(f'Class Distribution - {title}')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.savefig('class_distribution.eps', format='eps', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_feature_space(self, model, data_loader, device, model_name, n_samples=1000):
        model.eval()
        features = []
        labels = []
        
        with torch.no_grad():
            for i, (data, target) in enumerate(data_loader):
                if i * batch_size >= n_samples:
                    break
                data = data.to(device)
                
                if isinstance(model, EnhancedHybridModel):
                    # Get quantum-enhanced features
                    x = model.classical_net(data)
                    x = x.view(x.size(0), -1)
                    x = model.feature_reducer(x)
                    x = model.qlayer(x)  # Quantum transformed features
                else:
                    # Get deep classical features (before final layer)
                    x = model.conv_net(data)
                    x = x.view(x.size(0), -1)
                    if hasattr(model, 'head'):
                        for layer in list(model.head.children())[:-1]:
                            x = layer(x)
                
                features.append(x.cpu().numpy())
                labels.append(target.cpu().numpy())
        
        features = np.concatenate(features)[:n_samples]
        labels = np.concatenate(labels)[:n_samples]
        
        # Separate plots for PCA and t-SNE
        self._plot_pca(features, labels, model_name)
        self._plot_tsne(features, labels, model_name)
        self._plot_decision_boundaries(features, labels, model_name)
    
    def _plot_pca(self, features, labels, model_name):
        """Separate PCA plot with decision boundaries"""
        plt.figure(figsize=(12, 8))
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features)
        
        # Plot decision boundaries
        self._plot_decision_surface(features_pca, labels)
        
        scatter_pca = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, 
                                cmap='tab10', alpha=0.6, edgecolors='w', s=40)
        plt.title(f'{model_name} Feature Space - PCA\n'
                 f'Explained Variance: {pca.explained_variance_ratio_.sum():.2f}')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(scatter_pca, label='Class')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'feature_space_pca_{model_name}.eps', format='eps', 
                   bbox_inches='tight', dpi=300)
        plt.close()
    
    def _plot_tsne(self, features, labels, model_name):
        """Separate t-SNE plot with decision boundaries"""
        plt.figure(figsize=(12, 8))
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        features_tsne = tsne.fit_transform(features)
        
        # Plot decision boundaries
        self._plot_decision_surface(features_tsne, labels)
        
        scatter_tsne = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels,
                                 cmap='tab10', alpha=0.6, edgecolors='w', s=40)
        plt.title(f'{model_name} Feature Space - t-SNE')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter_tsne, label='Class')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'feature_space_tsne_{model_name}.eps', format='eps', 
                   bbox_inches='tight', dpi=300)
        plt.close()
    
    def _plot_decision_surface(self, X, y):
        """Helper to plot decision boundaries"""
        h = 0.02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Fit SVM to plot decision boundaries
        svm = SVC(kernel='rbf', gamma=2)
        svm.fit(X, y)
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundaries
        plt.contourf(xx, yy, Z, alpha=0.1, cmap='tab10')
        plt.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.5)
    
    def _plot_decision_boundaries(self, features, labels, model_name):
        """Separate plot showing class separation boundaries"""
        plt.figure(figsize=(12, 8))
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features)
        
        # Fit SVM and plot decision boundaries
        self._plot_decision_surface(features_pca, labels)
        
        # Plot data points
        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels,
                            cmap='tab10', alpha=0.8, edgecolors='w', s=40)
        
        # Calculate separation metrics
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(features, labels)
        separation_score = lda.score(features, labels)
        sil_score = silhouette_score(features, labels)
        
        plt.title(f'{model_name} Class Separation\n'
                 f'LDA Separation: {separation_score:.3f} | Silhouette Score: {sil_score:.3f}')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(scatter, label='Class')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'class_separation_{model_name}.eps', format='eps', 
                   bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_quantum_circuit(self, quantum_circuit, weights):
        dummy_input = torch.zeros(2**num_qubits)  # For amplitude encoding
        dummy_weights = torch.randn((num_layers, num_qubits))
        fig, ax = qml.draw_mpl(quantum_circuit)(dummy_input, dummy_weights)
        plt.title('Quantum Circuit Architecture')
        plt.savefig('quantum_circuit.eps', format='eps', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_metric_comparison(self):
        metrics = ['accuracy', 'f1', 'robustness', 'time']
        hybrid_metrics = []
        classical_metrics = []
        
        for m in metrics:
            if m in self.metrics_history['Hybrid'] and self.metrics_history['Hybrid'][m]:
                hybrid_metrics.append(np.mean(self.metrics_history['Hybrid'][m]))
            if m in self.metrics_history['Classical'] and self.metrics_history['Classical'][m]:
                classical_metrics.append(np.mean(self.metrics_history['Classical'][m]))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, hybrid_metrics, width, label='Hybrid')
        plt.bar(x + width/2, classical_metrics, width, label='Classical')
        
        plt.title('Model Performance Comparison')
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, axis='y')
        plt.savefig('metric_comparison.eps', format='eps', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_test_results(self):
        if not self.metrics_history['Hybrid_Test']['accuracy']:
            print("No test results to plot")
            return
            
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        hybrid_metrics = [self.metrics_history['Hybrid_Test'][m][-1] for m in metrics]
        classical_metrics = [self.metrics_history['Classical_Test'][m][-1] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, hybrid_metrics, width, label='Hybrid')
        plt.bar(x + width/2, classical_metrics, width, label='Classical')
        
        plt.title('Test Set Performance Comparison')
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, axis='y')
        plt.savefig('test_results.eps', format='eps', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_sample_images(self, data_loader, title, n_images=10):
        """Plot sample images from dataset"""
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        
        plt.figure(figsize=(15, 3))
        for i in range(n_images):
            plt.subplot(1, n_images, i+1)
            if images[i].shape[0] == 1:  # Grayscale
                plt.imshow(images[i].squeeze(), cmap='gray')
            else:  # RGB
                plt.imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
            plt.title(f"Label: {labels[i].item()}")
            plt.axis('off')
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(f'sample_images_{title.lower().replace(" ", "_")}.eps', 
                   format='eps', bbox_inches='tight', dpi=300)
        plt.close()
        
    def _plot_predictions(self, model, data_loader, device, model_name, n_images=10):
        """Plot sample predictions with true and predicted labels"""
        model.eval()
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        images, labels = images[:n_images].to(device), labels[:n_images].to(device)
        
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
        
        plt.figure(figsize=(15, 3))
        for i in range(n_images):
            plt.subplot(1, n_images, i+1)
            if images[i].shape[0] == 1:  # Grayscale
                plt.imshow(images[i].cpu().squeeze(), cmap='gray')
            else:  # RGB
                plt.imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))
            
            title_color = 'green' if preds[i] == labels[i] else 'red'
            plt.title(f"T:{labels[i].item()}\nP:{preds[i].item()}", color=title_color)
            plt.axis('off')
        
        plt.suptitle(f'Sample Predictions - {model_name}')
        plt.tight_layout()
        plt.savefig(f'predictions_{model_name}.eps', format='eps', bbox_inches='tight', dpi=300)
        plt.close()

def get_data_loaders(dataset_name, batch_size):
    config = dataset_config[dataset_name]
    
    # Find the Normalize transform in the composition
    normalize_transform = None
    for t in config['transform'].transforms:
        if isinstance(t, transforms.Normalize):
            normalize_transform = t
            break
    
    if not normalize_transform:
        raise ValueError("Normalize transform not found in the transform composition")
    
    # Full training set with augmentations
    if dataset_name == 'STL10':
        # STL10 has separate train and test splits
        train_data = config['loader'](
            './data', 
            split='train', 
            download=True,
            transform=config['transform']
        )
        val_data = config['loader'](
            './data',
            split='test',
            download=True,
            transform=config['transform']
        )
        # For STL10, we'll use the provided test split as our validation set
        # and create a smaller validation set from the training data
        train_size = int(0.8 * len(train_data))
        val_size = len(train_data) - train_size
        train_data, _ = random_split(train_data, [train_size, val_size])
    else:
        # For MNIST and CIFAR100
        train_data = config['loader'](
            './data', 
            train=True, 
            download=True,
            transform=config['transform']
        )
        # Split train into train and validation (80/20)
        train_size = int(0.8 * len(train_data))
        val_size = len(train_data) - train_size
        train_data, val_data = random_split(
            train_data, [train_size, val_size]
        )
    
    # Test set - completely unseen data with no augmentations
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_transform.mean, 
                           normalize_transform.std)
    ])
    
    if dataset_name == 'STL10':
        test_data = config['loader'](
            './data', 
            split='test', 
            transform=test_transform
        )
    else:
        test_data = config['loader'](
            './data', 
            train=False, 
            transform=test_transform
        )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader

def get_quantum_device():
    try:
        dev = qml.device("default.qubit", wires=num_qubits)
        print("Using default qubit simulator")
        return dev
    except Exception as e:
        print(f"Could not initialize quantum device: {str(e)}")
        raise

def quantum_circuit(inputs, weights):
    """Quantum circuit with amplitude encoding for 4 qubits (16 amplitudes)"""
    # Normalize inputs for amplitude encoding (must be positive and normalized)
    inputs = torch.abs(inputs)  # Ensure positive values
    inputs = inputs / torch.norm(inputs)  # Normalize to unit vector
    
    # Amplitude embedding of classical data (now needs 16 values)
    qml.AmplitudeEmbedding(inputs, wires=range(num_qubits), normalize=True, pad_with=0.)
    
    # Variational layers
    qml.BasicEntanglerLayers(weights, wires=range(num_qubits))
    
    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

class EnhancedHybridModel(nn.Module):
    def __init__(self, dataset_name):
        super().__init__()
        config = dataset_config[dataset_name]
        self.input_channels = config['input_channels']
        self.num_classes = config['num_classes']
        
        if dataset_name == 'MNIST':
            self.classical_net = nn.Sequential(
                nn.Conv2d(self.input_channels, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 4))
            )
            # Modified for 4-qubit amplitude encoding (output must be 2**4 = 16 values)
            self.feature_reducer = nn.Sequential(
                nn.Linear(32*4*4, 256),
                nn.ReLU(),
                nn.Linear(256, 16),  # 2^4 = 16 amplitudes needed
                nn.Softmax(dim=1)    # Ensure valid probability amplitudes
            )
        elif dataset_name == 'CIFAR100':
            self.classical_net = nn.Sequential(
                nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4))
            )
            # Modified for 4-qubit amplitude encoding
            self.feature_reducer = nn.Sequential(
                nn.Linear(128*4*4, 512),
                nn.ReLU(),
                nn.Linear(512, 16),  # 2^4 = 16 amplitudes needed
                nn.Softmax(dim=1)    # Ensure valid probability amplitudes
            )
        else:  # STL10
            self.classical_net = nn.Sequential(
                nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4))
            )
            # Modified for 4-qubit amplitude encoding
            self.feature_reducer = nn.Sequential(
                nn.Linear(256*4*4, 1024),
                nn.ReLU(),
                nn.Linear(1024, 16),  # 2^4 = 16 amplitudes needed
                nn.Softmax(dim=1)    # Ensure valid probability amplitudes
            )
        
        # Initialize quantum layer
        try:
            dev = get_quantum_device()
            qnode = qml.QNode(quantum_circuit, dev, interface="torch")
            weight_shapes = {"weights": (num_layers, num_qubits)}
            self.qlayer = TorchLayer(qnode, weight_shapes)
        except Exception as e:
            raise RuntimeError(f"Quantum layer initialization failed: {str(e)}")
        
        self.head = nn.Sequential(
            nn.Linear(num_qubits, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        x = self.classical_net(x)
        x = x.view(x.size(0), -1)
        x = self.feature_reducer(x)  # Now outputs 16 amplitudes
        q_out = self.qlayer(x)      # Amplitude-encoded quantum processing (4 qubits)
        return self.head(q_out)

class EnhancedClassicalCNN(nn.Module):
    def __init__(self, dataset_name):
        super().__init__()
        config = dataset_config[dataset_name]
        self.input_channels = config['input_channels']
        self.num_classes = config['num_classes']
        
        if dataset_name == 'MNIST':
            self.conv_net = nn.Sequential(
                nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 4))
            )
            self.head = nn.Sequential(
                nn.Linear(64*4*4, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, self.num_classes)
            )
        elif dataset_name == 'CIFAR100':
            self.conv_net = nn.Sequential(
                nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.head = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.num_classes)
            )
        else:  # STL10
            self.conv_net = nn.Sequential(
                nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.head = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, self.num_classes)
            )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

class ResourceTracker:
    def __init__(self):
        self.start_time = None
        self.start_cpu = None
        self.gpus = GPUtil.getGPUs() if torch.cuda.is_available() else []
    
    def start(self):
        self.start_time = time.time()
        self.start_cpu = psutil.cpu_percent(interval=None)
    
    def end(self):
        elapsed = time.time() - self.start_time
        cpu_usage = psutil.cpu_percent(interval=None) - self.start_cpu
        memory_usage = psutil.virtual_memory().used / (1024 ** 3)  # GB
        
        gpu_info = {}
        if self.gpus:
            for gpu in self.gpus:
                gpu_info[f'GPU_{gpu.id}_load'] = gpu.load
                gpu_info[f'GPU_{gpu.id}_mem'] = gpu.memoryUsed
        
        return {
            'time_sec': elapsed,
            'cpu_usage': cpu_usage,
            'memory_gb': memory_usage,
            **gpu_info
        }

def train_and_log(model, device, train_loader, optimizer, scheduler, epoch, model_type, resource_tracker, dataset_name, visualizer=None):
    model.train()
    running_loss = 0.0
    
    resource_tracker.start()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        if "Hybrid" in model_type:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        running_loss += loss.item()
    
    resources = resource_tracker.end()
    avg_train_loss = running_loss / len(train_loader)
    
    print(f"Epoch [{epoch}/{epochs}], {model_type} Train Loss: {avg_train_loss:.4f}")
    print(f"  Training Resources - Time: {resources['time_sec']:.2f}s, CPU: {resources['cpu_usage']:.1f}%, Memory: {resources['memory_gb']:.2f}GB")
    
    if visualizer:
        visualizer.update_metrics(model_type, {
            'train_loss': avg_train_loss,
            'time': resources['time_sec'],
            'cpu': resources['cpu_usage'],
            'memory': resources['memory_gb']
        })
    
    return avg_train_loss, resources

def test_and_log(model, device, test_loader, epoch, model_type, best_accuracy, dataset_name, visualizer=None):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.squeeze().cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    if correct > 0:
        avg_method = "macro" if epoch > 5 else "micro"
        precision = precision_score(all_targets, all_preds, average=avg_method, zero_division=0)
        recall = recall_score(all_targets, all_preds, average=avg_method, zero_division=0)
        f1 = f1_score(all_targets, all_preds, average=avg_method, zero_division=0)
    else:
        precision = recall = f1 = 0.0

    print(f"Epoch [{epoch}/{epochs}], {model_type} Val Loss: {test_loss:.4f}, "
          f"Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, "
          f"Recall: {recall:.4f}, F1: {f1:.4f}")

    if accuracy > best_accuracy:
        print(f"✨ {model_type} validation improved to {accuracy:.2f}%")
        best_accuracy = accuracy
        torch.save(model.state_dict(), f"best_{model_type.lower().replace(' ', '_')}_{dataset_name}.pth")

    if visualizer:
        visualizer.update_metrics(model_type, {
            'val_loss': test_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    return test_loss, accuracy, precision, recall, f1, best_accuracy

def evaluate_adversarial_robustness(model, data_loader, device, epsilon=0.1):
    model.eval()
    attack = PGD(model, eps=epsilon, alpha=0.01, steps=10)
    correct = 0
    total = 0
    
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        adv_data = attack(data, target)
        output = model(adv_data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    robustness = 100. * correct / total
    print(f"Adversarial Robustness (ε={epsilon}): {robustness:.2f}%")
    return robustness

def main(dataset_name='CIFAR100'):
    # Initialize visualizer with enhanced plotting capabilities
    visualizer = MLVisualizer()
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(dataset_name, batch_size)
    
    # 1. Plot sample images before training (new)
    print("\n===== VISUALIZING DATASET =====")
    visualizer.plot_sample_images(train_loader, 'Training Samples')
    visualizer.plot_sample_images(val_loader, 'Validation Samples')
    visualizer.plot_sample_images(test_loader, 'Test Samples')
    
    # 2. Plot class distributions before training
    visualizer.plot_class_distribution(train_loader.dataset, 'Training Set')
    visualizer.plot_class_distribution(val_loader.dataset, 'Validation Set')
    
    # Initialize models
    hybrid_model = EnhancedHybridModel(dataset_name).to(device)
    classical_model = EnhancedClassicalCNN(dataset_name).to(device)
    
    # Optimizers
    hybrid_optim = optim.AdamW(
        hybrid_model.parameters(),
        lr=0.001, weight_decay=0.01
    )
    
    classical_optim = optim.Adam(
        classical_model.parameters(),
        lr=0.003, weight_decay=0.005
    )
    
    # Schedulers
    hybrid_scheduler = OneCycleLR(
        hybrid_optim,
        max_lr=0.002,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3
    )
    
    classical_scheduler = CosineAnnealingWarmRestarts(
        classical_optim,
        T_0=5,
        T_mult=1,
        eta_min=1e-5
    )
    
    # Trackers
    hybrid_best = 0
    classical_best = 0
    resource_tracker = ResourceTracker()
    
    print("\n===== STARTING TRAINING =====")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print(f"Hybrid Model params: {sum(p.numel() for p in hybrid_model.parameters())}")
    print(f"Classical Model params: {sum(p.numel() for p in classical_model.parameters())}\n")

    best_hybrid_model = None
    best_classical_model = None

    for epoch in range(1, epochs+1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        
        # Hybrid training
        hybrid_train_loss, hybrid_resources = train_and_log(
            hybrid_model, device, train_loader, hybrid_optim, hybrid_scheduler, 
            epoch, "Hybrid", resource_tracker, dataset_name, visualizer
        )
        hybrid_val_loss, hybrid_acc, hybrid_prec, hybrid_rec, hybrid_f1, hybrid_best = test_and_log(
            hybrid_model, device, val_loader, epoch, "Hybrid", hybrid_best, dataset_name, visualizer
        )

        # Classical training
        classical_train_loss, classical_resources = train_and_log(
            classical_model, device, train_loader, classical_optim, classical_scheduler,
            epoch, "Classical", resource_tracker, dataset_name, visualizer
        )
        classical_val_loss, classical_acc, classical_prec, classical_rec, classical_f1, classical_best = test_and_log(
            classical_model, device, val_loader, epoch, "Classical", classical_best, dataset_name, visualizer
        )

        # Store best models
        if hybrid_acc == hybrid_best:
            best_hybrid_model = copy.deepcopy(hybrid_model)
        if classical_acc == classical_best:
            best_classical_model = copy.deepcopy(classical_model)

        # Adversarial evaluation (every 5 epochs)
        if epoch % 5 == 0:
            hybrid_robustness = evaluate_adversarial_robustness(hybrid_model, val_loader, device)
            classical_robustness = evaluate_adversarial_robustness(classical_model, val_loader, device)
            visualizer.update_metrics('Hybrid', {'robustness': hybrid_robustness})
            visualizer.update_metrics('Classical', {'robustness': classical_robustness})

        # Plot intermediate results every 10 epochs
        if epoch % 10 == 0:
            visualizer.plot_training_curves()
            visualizer.plot_resource_usage()

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Hybrid - Loss: {hybrid_train_loss:.4f}/{hybrid_val_loss:.4f}, Acc: {hybrid_acc:.2f}%")
        print(f"  Classical - Loss: {classical_train_loss:.4f}/{classical_val_loss:.4f}, Acc: {classical_acc:.2f}%")
        print(f"  Resources - Hybrid: {hybrid_resources['time_sec']:.2f}s, Classical: {classical_resources['time_sec']:.2f}s")
        
        # Early stopping check
        if epoch > 10:
            if dataset_name == 'MNIST' and hybrid_acc < 80 and classical_acc < 90:
                print("Early stopping due to poor performance")
                break
            elif dataset_name == 'CIFAR100' and hybrid_acc < 5 and classical_acc < 15:
                print("Early stopping due to poor performance")
                break
            elif dataset_name == 'STL10' and hybrid_acc < 30 and classical_acc < 40:
                print("Early stopping due to poor performance")
                break
        
        print("="*50 + "\n")

    # Final visualizations
    print("\n===== GENERATING FINAL VISUALIZATIONS =====")
    visualizer.plot_training_curves()
    visualizer.plot_resource_usage()
    visualizer.plot_metric_comparison()
    
    # Get class names
    if dataset_name == 'MNIST':
        class_names = [str(i) for i in range(10)]
    elif dataset_name == 'CIFAR100':
        class_names = [str(i) for i in range(100)]
    else:  # STL10
        class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 
                     'dog', 'horse', 'monkey', 'ship', 'truck']
    
    # Final evaluation on unseen test data
    print("\n===== FINAL TEST EVALUATION (UNSEEN DATA) =====")
    
    if best_hybrid_model:
        print("\nEvaluating Best Hybrid Model on Test Set:")
        test_loss, test_acc, test_prec, test_rec, test_f1, _ = test_and_log(
            best_hybrid_model, device, test_loader, 
            epoch, "Hybrid_Test", 0, dataset_name, visualizer
        )
        # Plot confusion matrix
        visualizer.plot_confusion_matrix(best_hybrid_model, test_loader, device, class_names, 'Hybrid')
        
        # Plot feature space with decision boundaries
        visualizer.plot_feature_space(best_hybrid_model, test_loader, device, 'Hybrid')
        
        # Plot sample predictions
        visualizer._plot_predictions(best_hybrid_model, test_loader, device, 'Hybrid')
    
    if best_classical_model:
        print("\nEvaluating Best Classical Model on Test Set:")
        test_loss, test_acc, test_prec, test_rec, test_f1, _ = test_and_log(
            best_classical_model, device, test_loader, 
            epoch, "Classical_Test", 0, dataset_name, visualizer
        )
        # Plot confusion matrix
        visualizer.plot_confusion_matrix(best_classical_model, test_loader, device, class_names, 'Classical')
        
        # Plot feature space with decision boundaries
        visualizer.plot_feature_space(best_classical_model, test_loader, device, 'Classical')
        
        # Plot sample predictions
        visualizer._plot_predictions(best_classical_model, test_loader, device, 'Classical')

    # Plot quantum circuit
    dummy_weights = torch.randn((num_layers, num_qubits))
    visualizer.plot_quantum_circuit(quantum_circuit, dummy_weights)
    
    # Plot final test results comparison
    visualizer.plot_test_results()

    print("\n===== FINAL RESULTS =====")
    print(f"Best Hybrid Validation Accuracy: {hybrid_best:.2f}%")
    print(f"Best Classical Validation Accuracy: {classical_best:.2f}%")

    # Save all plots
    print("\n===== GENERATED VISUALIZATIONS =====")
    print("1. Sample images before training (sample_images_*.eps)")
    print("2. Class distributions (class_distribution.eps)")
    print("3. Training curves (training_curves_*.eps)")
    print("4. Resource usage (resource_*.eps)")
    print("5. Metric comparisons (metric_comparison.eps)")
    print("6. Test results (test_results.eps)")
    print("7. Confusion matrices (confusion_matrix_*.eps)")
    print("8. Feature spaces (feature_space_*.eps)")
    print("9. Class separation boundaries (class_separation_*.eps)")
    print("10. Sample predictions (predictions_*.eps)")
    print("11. Quantum circuit (quantum_circuit.eps)")

    # Save model architectures
    with open('model_architectures.txt', 'w') as f:
        f.write("HYBRID MODEL ARCHITECTURE:\n")
        f.write(str(hybrid_model))
        f.write("\n\nCLASSICAL MODEL ARCHITECTURE:\n")
        f.write(str(classical_model))

if __name__ == "__main__":
    try:
        print("\n" + "="*60)
        print("TRAINING ON CIFAR100 DATASET")
        print("="*60)
        main(dataset_name='CIFAR100')
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nFinal troubleshooting steps:")
        print("1. Verify all variable names are consistent")
        print("2. Check all required parameters are passed between functions")
        print("3. For a clean environment:")
        print("   conda create -n qml python=3.9")
        print("   conda activate qml")
        print("   pip install pennylane==0.32 torch==2.0.1 torchvision")
