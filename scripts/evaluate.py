"""Evaluate a saved checkpoint."""
import os, sys, argparse, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from configs.config import DEVICE, EXPERIMENTS, DATASETS
from src.data.data_loader import get_data_loaders
from src.models.hybrid_model import EnhancedHybridModel
from src.models.classical_model import EnhancedClassicalCNN
from src.training.trainer import test_and_log
from src.visualization.visualizer import MLVisualizer


def main(exp_name, model_type, ckpt):
    cfg = EXPERIMENTS[exp_name]
    ds = cfg['dataset']
    out_dir = os.path.join(ROOT, 'results', exp_name)

    _, _, test_loader = get_data_loaders(ds, cfg['batch_size'])
    if model_type == 'hybrid':
        model = EnhancedHybridModel(ds, cfg['num_qubits'],
                                    cfg['num_layers'], cfg['encoding']).to(DEVICE)
        key, name = 'Hybrid_Test', 'Hybrid'
    else:
        model = EnhancedClassicalCNN(ds).to(DEVICE)
        key, name = 'Classical_Test', 'Classical'

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    viz = MLVisualizer(out_dir=out_dir, fmt=cfg['plot_format'])
    test_and_log(model, DEVICE, test_loader, 0, key, 0, ds, viz, cfg['epochs'])
    viz.plot_confusion_matrix(model, test_loader, DEVICE,
                              DATASETS[ds]['class_names'], name)
    viz._plot_predictions(model, test_loader, DEVICE, name)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', required=True, choices=list(EXPERIMENTS.keys()))
    p.add_argument('--model', default='hybrid', choices=['hybrid', 'classical'])
    p.add_argument('--checkpoint', required=True)
    a = p.parse_args()
    main(a.experiment, a.model, a.checkpoint)
