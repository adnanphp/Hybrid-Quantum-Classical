"""PGD adversarial robustness."""
from torchattacks import PGD


def evaluate_adversarial_robustness(model, data_loader, device, epsilon=0.1):
    model.eval()
    attack = PGD(model, eps=epsilon, alpha=0.01, steps=10)
    correct = total = 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        adv = attack(data, target)
        pred = model(adv).argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    robustness = 100. * correct / total
    print(f"Adversarial Robustness (eps={epsilon}): {robustness:.2f}%")
    return robustness
