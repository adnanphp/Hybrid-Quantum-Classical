# ⚛️ Hybrid Quantum-Classical Model for Image Classification

[![arXiv](https://img.shields.io/badge/arXiv-2509.13353-b31b1b.svg)](https://arxiv.org/abs/2509.13353)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-4B8BBE)]
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/adnanphp/Hybrid-Quantum-Classical/actions/workflows/ci.yml/badge.svg)](https://github.com/adnanphp/Hybrid-Quantum-Classical/actions)

> **A hybrid quantum-classical deep learning framework for image classification, combining classical convolutional networks with parameterized quantum circuits.**

This repository contains the **code, experimental outputs, figures, and research materials** for:

> **“Hybrid Quantum-Classical Model for Image Classification”**

The project investigates whether hybrid quantum-classical architectures can provide useful advantages in **classification accuracy, computational efficiency, model size, and adversarial robustness** compared with purely classical CNNs.

📄 **Paper:** [arXiv:2509.13353](https://arxiv.org/abs/2509.13353)

---

# 🧠 Research Overview

The project compares:

```text
┌───────────────────────────┐
│     Classical CNN         │
│                           │
│ Convolutional Layers      │
│        ↓                  │
│ Feature Extraction        │
│        ↓                  │
│ Classification            │
└───────────────────────────┘


             VS.


┌───────────────────────────┐
│ Hybrid Quantum-Classical  │
│                           │
│ Classical CNN             │
│        ↓                  │
│ Feature Compression       │
│        ↓                  │
│ Quantum Circuit            │
│        ↓                  │
│ Classical Classifier      │
└───────────────────────────┘
```

The experiments evaluate three major dimensions:

### 🎯 Performance

* Validation accuracy
* Test accuracy
* Classification performance across datasets

### ⚡ Efficiency

* Training time
* Parameter count
* CPU / memory usage

### 🛡️ Robustness

* Adversarial perturbations
* Robustness under **ε = 0.1**

---

# 📊 Key Results

## Benchmark Performance

| Dataset     | Classical Accuracy | Hybrid Accuracy |    Difference |
| ----------- | -----------------: | --------------: | ------------: |
| 🟢 MNIST    |         **98.21%** |      **99.38%** |  **+1.17 pp** |
| 🔵 CIFAR100 |         **32.25%** |      **41.69%** |  **+9.44 pp** |
| 🟣 STL10    |         **63.76%** |      **74.05%** | **+10.29 pp** |

> Results shown here correspond to the reported experimental configurations in the research paper.

---

# ⚡ Efficiency

The experiments also investigate computational efficiency.

### Reported findings

* 🚀 **5–12× faster training** in many experimental settings
* 📉 **6–32% fewer parameters**
* 💻 Lower CPU and memory usage in several configurations
* ⚛️ Quantum circuits are simulated rather than executed on physical quantum hardware

### Efficiency Pipeline

```text
                 Image
                   │
                   ▼
          ┌─────────────────┐
          │ Classical CNN   │
          └────────┬────────┘
                   │
                   ▼
          Feature Compression
                   │
                   ▼
          ┌─────────────────┐
          │ Quantum Circuit │
          │                 │
          │ Parameterized   │
          │ Quantum Gates   │
          └────────┬────────┘
                   │
                   ▼
          Quantum Measurements
                   │
                   ▼
          ┌─────────────────┐
          │ Classification  │
          └─────────────────┘
```

---

# 🛡️ Adversarial Robustness

Robustness experiments use adversarial perturbations with:

```text
ε = 0.1
```

The experiments compare the behavior of classical and hybrid models under perturbed inputs.

### Main observations

* Hybrid models show improved robustness on simpler datasets in the tested configurations.
* The difference becomes more comparable on higher-dimensional datasets.
* Robustness remains an important area for further investigation in hybrid quantum-classical learning.

---

# 🗃️ Datasets

The project evaluates the models on three image-classification benchmarks:

| Dataset         | Description                                     |
| --------------- | ----------------------------------------------- |
| 🟢 **MNIST**    | Handwritten digit classification                |
| 🔵 **CIFAR100** | 100-class natural image classification          |
| 🟣 **STL10**    | Natural image classification with larger images |

The datasets are used to evaluate how hybrid architectures behave as **image complexity and dimensionality increase**.

---

# 🔬 Experimental Setup

| Component        | Configuration                       |
| ---------------- | ----------------------------------- |
| Training epochs  | **50**                              |
| Datasets         | MNIST, CIFAR100, STL10              |
| Classical model  | CNN                                 |
| Hybrid model     | CNN + Parameterized Quantum Circuit |
| Quantum backend  | Simulated                           |
| Adversarial test | **ε = 0.1**                         |
| Framework        | PyTorch + PennyLane                 |
| Hardware         | CPU usable; GPU optional            |

---

# 🧰 Technology Stack

| Component              | Technology     |
| ---------------------- | -------------- |
| 🐍 Programming         | Python         |
| 🧠 Deep Learning       | PyTorch        |
| ⚛️ Quantum ML          | PennyLane      |
| 📊 Numerical Computing | NumPy          |
| 📈 Visualization       | Matplotlib     |
| 🔬 ML Utilities        | scikit-learn   |
| 📝 Research            | LaTeX          |
| 🔄 CI                  | GitHub Actions |

---

# 🚀 Getting Started

## Requirements

Python **3.8+** is required.

Install the main dependencies:

```bash
pip install torch torchvision pennylane matplotlib scikit-learn numpy
```

A GPU is optional and can accelerate classical CNN training.

> The quantum circuits in this repository are simulated, so the experiments can also be performed on a CPU.

---

# 📥 Installation

### 1. Clone the repository

```bash
git clone https://github.com/adnanphp/Hybrid-Quantum-Classical.git
cd Hybrid-Quantum-Classical
```

### 2. Install dependencies

```bash
pip install torch torchvision pennylane matplotlib scikit-learn numpy
```

### 3. Prepare the datasets

Use the preprocessing scripts in:

```text
data/
```

or download the required benchmark datasets.

Expected dataset directories may include:

```text
data/
├── MNIST/
├── CIFAR100/
└── STL10/
```

---

# 🏋️ Training

## Classical CNN

Example:

```bash
python codes/train_classical.py \
    --dataset MNIST \
    --epochs 50
```

## Hybrid Quantum-Classical Model

```bash
python codes/train_hybrid.py \
    --dataset MNIST \
    --epochs 50
```

The same workflow can be adapted for:

```text
MNIST
CIFAR100
STL10
```

---

# 🧪 Evaluation

Evaluate a trained model using:

```bash
python codes/evaluate.py \
    --model hybrid \
    --dataset CIFAR100
```

Evaluation outputs include model performance metrics and experimental results.

---

# 📊 Results & Visualization

Generated outputs are organized into:

```text
outputs/
```

and:

```text
figures/
```

### Outputs may include

* Training logs
* Validation metrics
* Test metrics
* Accuracy curves
* Loss curves
* Confusion matrices
* Robustness results
* Feature visualizations

---

# 📁 Repository Structure

```text
Hybrid-Quantum-Classical/
│
├── README.md
├── LICENSE
├── references.bib
│
├── codes/
│   ├── train_classical.py
│   ├── train_hybrid.py
│   ├── evaluate.py
│   └── utilities/
│
├── data/
│   ├── MNIST/
│   ├── CIFAR100/
│   └── STL10/
│
├── figures/
│   ├── accuracy/
│   ├── loss/
│   ├── confusion_matrices/
│   └── visualizations/
│
├── outputs/
│   ├── logs/
│   ├── metrics/
│   └── experiment_results/
│
└── report/
    ├── *.tex
    └── *.pdf
```

---

# 🔄 Research Workflow

```text
                 ┌─────────────────┐
                 │    Dataset      │
                 │ MNIST/CIFAR/STL │
                 └────────┬────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Data Preprocessing│
                └─────────┬─────────┘
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
    ┌─────────────────┐       ┌─────────────────────┐
    │ Classical CNN   │       │ Hybrid Architecture │
    │                 │       │                     │
    │ Conv Layers     │       │ Conv Layers         │
    │       ↓         │       │       ↓             │
    │ Classifier      │       │ Quantum Circuit     │
    └────────┬────────┘       │       ↓             │
             │                │ Classifier          │
             │                └──────────┬──────────┘
             │                           │
             └─────────────┬─────────────┘
                           ▼
                  ┌─────────────────┐
                  │    Evaluation   │
                  └────────┬────────┘
                           │
             ┌─────────────┼──────────────┐
             ▼             ▼              ▼
        Accuracy       Efficiency     Robustness
```

---

# 📌 Main Findings

The experiments indicate that the tested hybrid quantum-classical configurations can achieve competitive or higher classification accuracy than their classical counterparts while using fewer parameters and, in several settings, less training time.

The magnitude of the observed differences varies across datasets, highlighting the importance of **dataset complexity, architecture design, and experimental configuration** when evaluating hybrid quantum-classical models.

---

# 🔮 Future Work

Several directions remain open for further research:

### ⚛️ Real Quantum Hardware

Move beyond simulation and evaluate the models on physical quantum processors.

### 🧠 Deeper Quantum Architectures

Investigate:

* More expressive parameterized circuits
* Larger numbers of qubits
* Alternative quantum feature maps
* Different entanglement strategies

### 🌐 Additional Applications

Extend the approach to:

* Natural language processing
* Time-series analysis
* Scientific datasets
* Multimodal learning

### 🛡️ Robustness

Further investigate adversarial robustness, particularly for high-dimensional and complex datasets.

---

# 📄 Research Paper

This repository accompanies the paper:

> **Muhammad Adnan Shahzad.**
> *Hybrid Quantum-Classical Model for Image Classification.*
> arXiv:2509.13353, submitted September 14, 2025.

📚 **Paper:**
https://arxiv.org/abs/2509.13353

---

# 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{shahzad2025hybrid,
  title   = {Hybrid Quantum-Classical Model for Image Classification},
  author  = {Shahzad, Muhammad Adnan},
  journal = {arXiv preprint arXiv:2509.13353},
  year    = {2025}
}
```

---

# 👤 Author

**Muhammad Adnan Shahzad**

Research interests:

```text
Machine Learning
Deep Learning
Computer Vision
Quantum Machine Learning
Scientific Machine Learning
AI Research
```

---

# 📜 License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

<p align="center">

⚛️ <b>Hybrid Quantum-Classical Learning</b>

<br>

PyTorch · PennyLane · Quantum Machine Learning · Computer Vision

<br><br>

<i>Exploring the intersection of quantum computing and modern deep learning.</i>

</p>
