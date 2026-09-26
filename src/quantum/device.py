"""PennyLane device factory."""
import pennylane as qml


def get_quantum_device(num_qubits):
    try:
        dev = qml.device("default.qubit", wires=num_qubits)
        print(f"Using default.qubit simulator with {num_qubits} qubits")
        return dev
    except Exception as e:
        print(f"Could not initialize quantum device: {e}")
        raise
