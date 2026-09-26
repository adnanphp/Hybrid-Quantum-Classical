"""
Single training entry point for every experiment.

Usage:
    python scripts/train.py --experiment mnist_4q
    python scripts/train.py --experiment cifar100_4q
    python scripts/train.py --experiment stl10_4q
    python scripts/train.py --experiment mnist_2q_angle
"""
import os
import sys
import copy
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

# Make project root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from configs.config import DEVICE, EXPERIMENTS
from src.data.data_loader import get_data_loaders
from src.models.hybrid_model import EnhancedHybridModel
from src.models.classical_model import EnhancedClassicalCNN
from src.training.trainer import train_and_log, test_and_log
from src.utils.resource_tracker import ResourceTracker
from src.utils.adversarial import evaluate_adversarial_robustness
from src.visualization.visualizer import MLVisualizer
from src.quantum.encodings import CIRCUITS


def run(exp_name):
    cfg = EXPERIMENTS[exp_name]
    dataset_name = cfg['dataset']
    out_dir = os.path.join(ROOT, 'results', exp_name)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {exp_name}")
    print(f"  dataset={dataset_name}  qubits={cfg['num_qubits']}  "
          f"layers={cfg['num_layers']}  encoding={cfg['encoding']}")
    print(f"  epochs={cfg['epochs']}  batch_size={cfg['batch_size']}")
    print(f"  results -> {out_dir}")
    print("=" * 70)

    visualizer = MLVisualizer(out_dir=out_dir, fmt=cfg['plot_format'])
    visualizer.num_qubits = cfg['num_qubits']
    visualizer.num_layers = cfg['num_layers']

    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_name, cfg['batch_size'])

    # Dataset sanity plots
    visualizer.plot_sample_images(train_loader, 'Training Samples')
    visualizer.plot_sample_images(val_loader,   'Validation Samples')
    visualizer.plot_sample_images(test_loader,  'Test Samples')
    visualizer.plot_class_distribution(train_loader.dataset, 'Training Set')
    visualizer.plot_class_distribution(val_loader.dataset,   'Validation Set')

    # Models
    hybrid_model = EnhancedHybridModel(
        dataset_name,
        num_qubits=cfg['num_qubits'],
        num_layers=cfg['num_layers'],
        encoding=cfg['encoding'],
    ).to(DEVICE)
    classical_model = EnhancedClassicalCNN(dataset_name).to(DEVICE)

    hybrid_optim    = optim.AdamW(hybrid_model.parameters(),
                                  lr=cfg['lr_hybrid'], weight_decay=0.01)
    classical_optim = optim.Adam(classical_model.parameters(),
                                 lr=cfg['lr_classical'], weight_decay=0.005)

    hybrid_scheduler = OneCycleLR(
        hybrid_optim, max_lr=cfg['max_lr_hybrid'],
        steps_per_epoch=len(train_loader),
        epochs=cfg['epochs'], pct_start=0.3)
    classical_scheduler = CosineAnnealingWarmRestarts(
        classical_optim, T_0=5, T_mult=1, eta_min=1e-5)

    hybrid_best = classical_best = 0
    best_hybrid_model = best_classical_model = None
    resource_tracker = ResourceTracker()

    print(f"Device: {DEVICE}")
    print(f"Hybrid params:    {sum(p.numel() for p in hybrid_model.parameters())}")
    print(f"Classical params: {sum(p.numel() for p in classical_model.parameters())}\n")

    for epoch in range(1, cfg['epochs'] + 1):
        print(f"\n=== Epoch {epoch}/{cfg['epochs']} ===")

        h_train_loss, h_res = train_and_log(
            hybrid_model, DEVICE, train_loader, hybrid_optim, hybrid_scheduler,
            epoch, "Hybrid", resource_tracker, dataset_name, visualizer,
            cfg['epochs'], out_dir)
        h_val_loss, h_acc, _, _, _, hybrid_best = test_and_log(
            hybrid_model, DEVICE, val_loader, epoch, "Hybrid",
            hybrid_best, dataset_name, visualizer, cfg['epochs'], out_dir)

        c_train_loss, c_res = train_and_log(
            classical_model, DEVICE, train_loader, classical_optim, classical_scheduler,
            epoch, "Classical", resource_tracker, dataset_name, visualizer,
            cfg['epochs'], out_dir)
        c_val_loss, c_acc, _, _, _, classical_best = test_and_log(
            classical_model, DEVICE, val_loader, epoch, "Classical",
            classical_best, dataset_name, visualizer, cfg['epochs'], out_dir)

        if h_acc == hybrid_best:    best_hybrid_model = copy.deepcopy(hybrid_model)
        if c_acc == classical_best: best_classical_model = copy.deepcopy(classical_model)

        if epoch % 5 == 0:
            hr = evaluate_adversarial_robustness(hybrid_model, val_loader, DEVICE)
            cr = evaluate_adversarial_robustness(classical_model, val_loader, DEVICE)
            visualizer.update_metrics('Hybrid',    {'robustness': hr})
            visualizer.update_metrics('Classical', {'robustness': cr})

        if epoch % 10 == 0:
            visualizer.plot_training_curves()
            visualizer.plot_resource_usage()

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Hybrid    Loss: {h_train_loss:.4f}/{h_val_loss:.4f}  Acc: {h_acc:.2f}%")
        print(f"  Classical Loss: {c_train_loss:.4f}/{c_val_loss:.4f}  Acc: {c_acc:.2f}%")
        print(f"  Time  H: {h_res['time_sec']:.2f}s  C: {c_res['time_sec']:.2f}s")
        print("=" * 50)

        # Early stopping
        if epoch > 10:
            min_h, min_c = cfg['early_stop']
            if h_acc < min_h and c_acc < min_c:
                print("Early stopping due to poor performance"); break

    # ---- Final report ----
    visualizer.plot_training_curves()
    visualizer.plot_resource_usage()
    visualizer.plot_metric_comparison()

    class_names = __import__('configs.config', fromlist=['DATASETS']).DATASETS[dataset_name]['class_names']

    if best_hybrid_model:
        print("\nFinal test - Hybrid:")
        test_and_log(best_hybrid_model, DEVICE, test_loader, epoch,
                     "Hybrid_Test", 0, dataset_name, visualizer,
                     cfg['epochs'], out_dir)
        visualizer.plot_confusion_matrix(best_hybrid_model, test_loader,
                                         DEVICE, class_names, 'Hybrid')
        visualizer.plot_feature_space(best_hybrid_model, test_loader,
                                      DEVICE, 'Hybrid')
        visualizer._plot_predictions(best_hybrid_model, test_loader,
                                     DEVICE, 'Hybrid')

    if best_classical_model:
        print("\nFinal test - Classical:")
        test_and_log(best_classical_model, DEVICE, test_loader, epoch,
                     "Classical_Test", 0, dataset_name, visualizer,
                     cfg['epochs'], out_dir)
        visualizer.plot_confusion_matrix(best_classical_model, test_loader,
                                         DEVICE, class_names, 'Classical')
        visualizer.plot_feature_space(best_classical_model, test_loader,
                                      DEVICE, 'Classical')
        visualizer._plot_predictions(best_classical_model, test_loader,
                                     DEVICE, 'Classical')

    dummy_w = torch.randn((cfg['num_layers'], cfg['num_qubits']))
    visualizer.plot_quantum_circuit(CIRCUITS[cfg['encoding']], dummy_w)
    visualizer.plot_test_results()

    print(f"\nBest Hybrid Val Acc:    {hybrid_best:.2f}%")
    print(f"Best Classical Val Acc: {classical_best:.2f}%")
    print(f"All artefacts saved in: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', default='mnist_4q',
                   choices=list(EXPERIMENTS.keys()))
    args = p.parse_args()
    try:
        run(args.experiment)
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
