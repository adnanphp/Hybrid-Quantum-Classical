"""Quantum circuits for amplitude and angle encoding."""
import numpy as np
import torch
import pennylane as qml


# ---------------------------------------------------------------------------
# 4-qubit amplitude encoding (used by MNIST/CIFAR100/STL10 4-qubit variants)
# ---------------------------------------------------------------------------
def amplitude_circuit(inputs, weights):
    """Amplitude embedding. inputs must be a 2**n vector."""
    inputs = torch.abs(inputs)
    inputs = inputs / torch.norm(inputs)

    qml.AmplitudeEmbedding(inputs, wires=range(len(inputs).bit_length() - 1),
                           normalize=True, pad_with=0.)
    qml.BasicEntanglerLayers(weights, wires=range(weights.shape[1]))

    return [qml.expval(qml.PauliZ(i)) for i in range(weights.shape[1])]


# ---------------------------------------------------------------------------
# 2-qubit angle encoding (used by MNIST 2-qubit variant)
# ---------------------------------------------------------------------------
def angle_circuit(inputs, weights):
    inputs = (inputs / torch.max(torch.abs(inputs))) * np.pi
    qml.AngleEmbedding(inputs, wires=range(len(inputs)), rotation='Y')
    qml.BasicEntanglerLayers(weights, wires=range(weights.shape[1]))
    return [qml.expval(qml.PauliZ(i)) for i in range(weights.shape[1])]


CIRCUITS = {
    'amplitude': amplitude_circuit,
    'angle': angle_circuit,
}
