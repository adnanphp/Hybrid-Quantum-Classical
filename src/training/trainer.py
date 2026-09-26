"""Train / test loops."""
import torch
import torch.nn as nn
from src.training.metrics import compute_metrics

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


def train_and_log(model, device, train_loader, optimizer, scheduler, epoch,
                  model_type, resource_tracker, dataset_name, visualizer,
                  epochs, out_dir='.'):
    model.train()
    running_loss = 0.0
    resource_tracker.start()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        if "Hybrid" in model_type:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        running_loss += loss.item()
    resources = resource_tracker.end()
    avg = running_loss / len(train_loader)
    print(f"Epoch [{epoch}/{epochs}], {model_type} Train Loss: {avg:.4f}")
    print(f"  Resources - Time: {resources['time_sec']:.2f}s, "
          f"CPU: {resources['cpu_usage']:.1f}%, Mem: {resources['memory_gb']:.2f}GB")
    visualizer.update_metrics(model_type, {
        'train_loss': avg,
        'time': resources['time_sec'],
        'cpu': resources['cpu_usage'],
        'memory': resources['memory_gb'],
    })
    return avg, resources


def test_and_log(model, device, test_loader, epoch, model_type,
                 best_accuracy, dataset_name, visualizer, epochs, out_dir='.'):
    model.eval()
    test_loss = correct = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            test_loss += criterion(out, target).item()
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.squeeze().cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    p, r, f = compute_metrics(all_targets, all_preds, epoch)
    print(f"Epoch [{epoch}/{epochs}], {model_type} Val Loss: {test_loss:.4f}, "
          f"Acc: {acc:.2f}%, P: {p:.4f}, R: {r:.4f}, F1: {f:.4f}")
    if acc > best_accuracy:
        print(f"*** {model_type} val improved to {acc:.2f}%")
        best_accuracy = acc
        torch.save(model.state_dict(),
                   f"best_{model_type.lower().replace(' ', '_')}_{dataset_name}.pth")
    visualizer.update_metrics(model_type, {
        'val_loss': test_loss, 'accuracy': acc,
        'precision': p, 'recall': r, 'f1': f,
    })
    return test_loss, acc, p, r, f, best_accuracy
