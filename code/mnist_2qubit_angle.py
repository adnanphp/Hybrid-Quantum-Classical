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

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_qubits = 2  # Reduced to 2 for stability
num_layers = 2
batch_size = 64
epochs = 10

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
    }
}

# Visualization Class
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
        # Plot loss curves
        self._plot_single_metric(
            metric_keys=['train_loss', 'val_loss'],
            labels=['Training Loss', 'Validation Loss'],
            colors=['b-', 'r-', 'b--', 'r--'],
            ylabel='Loss',
            filename=f'{base_filename}_loss.png'
        )
        
        # Plot accuracy
        self._plot_single_metric(
            metric_keys=['accuracy'],
            labels=['Accuracy'],
            colors=['g-', 'm-'],
            ylabel='Accuracy (%)',
            filename=f'{base_filename}_accuracy.png'
        )
        
        # Plot F1 score
        self._plot_single_metric(
            metric_keys=['f1'],
            labels=['F1 Score'],
            colors=['c-', 'y-'],
            ylabel='F1 Score',
            filename=f'{base_filename}_f1.png'
        )
        
        # Plot robustness if available
        if any(self.metrics_history['Hybrid']['robustness']):
            self._plot_single_metric(
                metric_keys=['robustness'],
                labels=['Robustness'],
                colors=['k-', 'k--'],
                ylabel='Robustness (%)',
                filename=f'{base_filename}_robustness.png',
                xvals=range(5, len(self.metrics_history['Hybrid']['robustness'])*5 +1, 5)
            )

    def _plot_single_metric(self, metric_keys, labels, colors, ylabel, filename, xvals=None):
        """Helper function to plot a single metric comparison"""
        plt.figure(figsize=(12, 6))
        
        for i, metric in enumerate(metric_keys):
            # Plot hybrid model
            if metric in self.metrics_history['Hybrid']:
                y_values = self.metrics_history['Hybrid'][metric]
                x_values = xvals if xvals is not None else range(1, len(y_values)+1)
                plt.plot(x_values, y_values, colors[i*2], 
                        label=f'Hybrid {labels[i]}')
            
            # Plot classical model
            if metric in self.metrics_history['Classical']:
                y_values = self.metrics_history['Classical'][metric]
                x_values = xvals if xvals is not None else range(1, len(y_values)+1)
                plt.plot(x_values, y_values, colors[i*2+1], 
                        label=f'Classical {labels[i]}')
        
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} Comparison')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_resource_usage(self, filename='resource_usage.png'):
        plt.figure(figsize=(18, 6))
        
        # Time comparison
        plt.subplot(1, 3, 1)
        plt.plot(self.metrics_history['Hybrid']['time'], 'b-', label='Hybrid')
        plt.plot(self.metrics_history['Classical']['time'], 'r-', label='Classical')
        plt.title('Training Time per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Time (s)')
        plt.legend()
        plt.grid(True)
        
        # CPU usage
        plt.subplot(1, 3, 2)
        plt.plot(self.metrics_history['Hybrid']['cpu'], 'b-', label='Hybrid')
        plt.plot(self.metrics_history['Classical']['cpu'], 'r-', label='Classical')
        plt.title('CPU Usage per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('CPU Usage (%)')
        plt.legend()
        plt.grid(True)
        
        # Memory usage
        plt.subplot(1, 3, 3)
        plt.plot(self.metrics_history['Hybrid']['memory'], 'b-', label='Hybrid')
        plt.plot(self.metrics_history['Classical']['memory'], 'r-', label='Classical')
        plt.title('Memory Usage per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Memory (GB)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
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
        plt.savefig(f'confusion_matrix_{model_name}.png', format='png', bbox_inches='tight', dpi=300)
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
        plt.savefig('class_distribution.png', format='png', bbox_inches='tight', dpi=300)
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
        
        # Create figure with PCA and t-SNE side by side
        plt.figure(figsize=(24, 10))
        
        # PCA Visualization
        plt.subplot(1, 2, 1)
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features)
        scatter_pca = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, 
                                cmap='tab10', alpha=0.6)
        plt.title(f'{model_name} Feature Space - PCA\n'
                 f'Explained Variance: {pca.explained_variance_ratio_.sum():.2f}')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(scatter_pca, label='Class')
        
        # t-SNE Visualization
        plt.subplot(1, 2, 2)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        features_tsne = tsne.fit_transform(features)
        scatter_tsne = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels,
                                 cmap='tab10', alpha=0.6)
        plt.title(f'{model_name} Feature Space - t-SNE')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter_tsne, label='Class')
        
        # Calculate and display separation metrics
        if len(np.unique(labels)) > 1:
            lda = LinearDiscriminantAnalysis(n_components=1)
            lda.fit(features, labels)
            separation_score = lda.score(features, labels)
            sil_score = silhouette_score(features, labels)
            
            plt.figtext(0.5, 0.01, 
                       f'LDA Separation: {separation_score:.3f} | Silhouette Score: {sil_score:.3f}',
                       ha='center', fontsize=12,
                       bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'feature_space_{model_name}.png', format='png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_quantum_circuit(self, quantum_circuit, weights):
        dummy_input = torch.zeros(num_qubits)  # Match number of qubits
        dummy_weights = torch.randn((num_layers, num_qubits))  # Correct shape for layers
        fig, ax = qml.draw_mpl(quantum_circuit)(dummy_input, dummy_weights)
        plt.title('Quantum Circuit Architecture')
        plt.savefig('quantum_circuit.png', format='png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_metric_comparison(self, filename='metric_comparison.png'):
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
        plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_test_results(self, filename='test_results.png'):
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
        plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
        plt.close()

# [Rest of your existing code remains exactly the same...]
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
    # Normalize inputs to [-π, π] range
    inputs = (inputs / torch.max(torch.abs(inputs))) * np.pi
    qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation='Y')
    qml.BasicEntanglerLayers(weights, wires=range(num_qubits))
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
            self.feature_reducer = nn.Sequential(
                nn.Linear(32*4*4, 128),
                nn.ReLU(),
                nn.Linear(128, num_qubits),
                nn.Tanh()
            )
        else:  # CIFAR100
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
            self.feature_reducer = nn.Sequential(
                nn.Linear(128*4*4, 256),
                nn.ReLU(),
                nn.Linear(256, num_qubits),
                nn.Tanh()
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
        x = self.feature_reducer(x)
        q_out = self.qlayer(x)
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
        else:  # CIFAR100
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

def main(dataset_name='MNIST'):
    # Initialize visualizer
    visualizer = MLVisualizer()
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(dataset_name, batch_size)
    
    # Plot class distributions
    visualizer.plot_class_distribution(train_loader.dataset, 'Training Set')
    visualizer.plot_class_distribution(val_loader.dataset, 'Validation Set')
    
    # Initialize models
    hybrid_model = EnhancedHybridModel(dataset_name).to(device)
    classical_model = EnhancedClassicalCNN(dataset_name).to(device)
    
    # Optimizers
    hybrid_optim = optim.AdamW(
        hybrid_model.parameters(),
        lr=0.0005 if dataset_name == 'CIFAR100' else 0.001,
        weight_decay=0.01
    )
    
    classical_optim = optim.Adam(
        classical_model.parameters(),
        lr=0.001 if dataset_name == 'CIFAR100' else 0.003,
        weight_decay=0.005
    )
    
    # Schedulers
    hybrid_scheduler = OneCycleLR(
        hybrid_optim,
        max_lr=0.001 if dataset_name == 'CIFAR100' else 0.002,
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
        
        print("="*50 + "\n")

    # Final visualizations
    visualizer.plot_training_curves()
    visualizer.plot_resource_usage()
    visualizer.plot_metric_comparison()
    
    # Get class names
    if dataset_name == 'MNIST':
        class_names = [str(i) for i in range(10)]
    elif dataset_name == 'CIFAR100':
        class_names = [str(i) for i in range(100)]  # Would be better with actual class names
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
        visualizer.plot_confusion_matrix(best_hybrid_model, test_loader, device, class_names, 'Hybrid')
        visualizer.plot_feature_space(best_hybrid_model, test_loader, device, 'Hybrid')
    
    if best_classical_model:
        print("\nEvaluating Best Classical Model on Test Set:")
        test_loss, test_acc, test_prec, test_rec, test_f1, _ = test_and_log(
            best_classical_model, device, test_loader, 
            epoch, "Classical_Test", 0, dataset_name, visualizer
        )
        visualizer.plot_confusion_matrix(best_classical_model, test_loader, device, class_names, 'Classical')
        visualizer.plot_feature_space(best_classical_model, test_loader, device, 'Classical')

    # Plot quantum circuit
    dummy_weights = torch.randn((num_layers, num_qubits))
    visualizer.plot_quantum_circuit(quantum_circuit, dummy_weights)
    
    # Plot final test results comparison
    visualizer.plot_test_results()

    print("\n===== FINAL RESULTS =====")
    print(f"Best Hybrid Validation Accuracy: {hybrid_best:.2f}%")
    print(f"Best Classical Validation Accuracy: {classical_best:.2f}%")

    # Save all plots
    print("\n===== GENERATED PLOTS =====")
    print("1. Training curves (training_curves.png)")
    print("2. Resource usage (resource_usage.png)")
    print("3. Metric comparison (metric_comparison.png)")
    print("4. Test results (test_results.png)")
    print("5. Class distribution (class_distribution.png)")
    print("6. Confusion matrices (confusion_matrix_*.png)")
    print("7. Feature space visualizations (feature_space_*.png)")
    print("8. Quantum circuit (quantum_circuit.png)")

if __name__ == "__main__":
    try:
        print("\n" + "="*60)
        print("TRAINING ON MNIST DATASET")
        print("="*60)
        main(dataset_name='MNIST')
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nFinal troubleshooting steps:")
        print("1. Verify all variable names are consistent")
        print("2. Check all required parameters are passed between functions")
        print("3. For a clean environment:")
        print("   conda create -n qml python=3.9")
        print("   conda activate qml")
        print("   pip install pennylane==0.32 torch==2.0.1 torchvision")
