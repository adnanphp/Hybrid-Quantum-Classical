"""All plotting utilities, parameterised by output dir + format."""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.svm import SVC
import pennylane as qml

from src.models.hybrid_model import EnhancedHybridModel


class MLVisualizer:
    def __init__(self, out_dir='.', fmt='eps'):
        self.out_dir = out_dir
        self.fmt = fmt
        os.makedirs(out_dir, exist_ok=True)

        self.batch_size = 64  # used by plot_feature_space
        self.metrics_history = {
            'Hybrid': {'train_loss': [], 'val_loss': [], 'accuracy': [],
                       'precision': [], 'recall': [], 'f1': [], 'robustness': [],
                       'time': [], 'cpu': [], 'memory': []},
            'Classical': {'train_loss': [], 'val_loss': [], 'accuracy': [],
                          'precision': [], 'recall': [], 'f1': [], 'robustness': [],
                          'time': [], 'cpu': [], 'memory': []},
            'Hybrid_Test':    {'val_loss': [], 'accuracy': [], 'precision': [],
                               'recall': [], 'f1': []},
            'Classical_Test': {'val_loss': [], 'accuracy': [], 'precision': [],
                               'recall': [], 'f1': []},
        }

    # -- helper used by every plot method --
    def _save(self, name):
        plt.savefig(os.path.join(self.out_dir, f"{name}.{self.fmt}"),
                    format=self.fmt, bbox_inches='tight', dpi=300)
        plt.close()

    def update_metrics(self, model_type, metrics_dict):
        if model_type not in self.metrics_history:
            print(f"Warning: unknown model type '{model_type}'")
            return
        for k, v in metrics_dict.items():
            if k in self.metrics_history[model_type]:
                self.metrics_history[model_type][k].append(v)
            else:
                print(f"Warning: unknown metric '{k}' for '{model_type}'")

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
        self._save(f'{base_filename}_loss.eps')
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
        self._save(f'{base_filename}_accuracy.eps')
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
        self._save(f'{base_filename}_f1.eps')
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
            self._save(f'{base_filename}_robustness.eps')
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
        self._save('resource_time.eps')
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
        self._save('resource_cpu.eps')
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
        self._save('resource_memory.eps')
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
        self._save(f'confusion_matrix_{model_name}.eps')
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
        self._save('class_distribution.eps')
        plt.close()
    
    def plot_feature_space(self, model, data_loader, device, model_name, n_samples=1000):
        model.eval()
        features = []
        labels = []
        
        with torch.no_grad():
            for i, (data, target) in enumerate(data_loader):
                if i * self.batch_size >= n_samples:
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
        self._save(f'feature_space_pca_{model_name}.eps')
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
        self._save(f'feature_space_tsne_{model_name}.eps')
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
        self._save(f'class_separation_{model_name}.eps')
        plt.close()
    
    def plot_quantum_circuit(self, quantum_circuit, weights):
        dummy_input = torch.zeros(2**num_qubits)  # For amplitude encoding
        dummy_weights = torch.randn((num_layers, num_qubits))
        fig, ax = qml.draw_mpl(quantum_circuit)(dummy_input, dummy_weights)
        plt.title('Quantum Circuit Architecture')
        self._save('quantum_circuit.eps')
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
        self._save('metric_comparison.eps')
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
        self._save('test_results.eps')
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
        self._save(f'sample_images_{title.lower().replace(" ", "_")}.eps')
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
        self._save(f'predictions_{model_name}.eps')
        plt.close()
