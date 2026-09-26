"""Quantum circuit definition and device helpers."""
import torch
import pennylane as qml
from configs.config import num_qubits, num_layers


def get_quantum_device():
    """Return a PennyLane simulator device."""
    try:
        dev = qml.device("default.qubit", wires=num_qubits)
        print("Using default qubit simulator")
        return dev
    except Exception as e:
        print(f"Could not initialize quantum device: {str(e)}")
        raise


def quantum_circuit(inputs, weights):
    """Quantum circuit with amplitude encoding for 4 qubits (16 amplitudes)."""
    inputs = torch.abs(inputs)
    inputs = inputs / torch.norm(inputs)

    qml.AmplitudeEmbedding(inputs, wires=range(num_qubits),
                           normalize=True, pad_with=0.)
    qml.BasicEntanglerLayers(weights, wires=range(num_qubits))

    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
