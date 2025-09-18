# Hybrid-Quantum-Classical
[![arXiv](https://img.shields.io/badge/arXiv-2509.13353-b31b1b.svg)](https://arxiv.org/abs/2509.13353)  
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Build Status](https://github.com/adnanphp/Hybrid-Quantum-Classical/actions/workflows/ci.yml/badge.svg)](https://github.com/adnanphp/Hybrid-Quantum-Classical/actions)  

---

## 📚 Citation

If you use this work in your research, please cite:

> Muhammad Adnan Shahzad. *Hybrid Quantum-Classical Model for Image Classification*. arXiv:2509.13353, submitted 14 Sep 2025. ([arxiv.org](https://arxiv.org/abs/2509.13353))

---

## 🧠 Project Overview

This repo contains all code, data, figures, and output logs for the paper **“Hybrid Quantum-Classical Model for Image Classification”**. We compare hybrid quantum-classical neural networks with purely classical convolutional neural networks (CNNs) across three benchmark datasets (MNIST, CIFAR100, STL10) to evaluate:

- **Performance** (validation & test accuracy)  
- **Efficiency** (training time, resource usage)  
- **Robustness** (adversarial perturbations with ε = 0.1)  

---

## 🔍 Key Results

| Dataset   | Classical Accuracy | Hybrid Accuracy | Approximate Gains / Notes |
|------------|---------------------|------------------|-----------------------------|
| MNIST     | ~98.21%             | ~99.38%          | Hybrid outperforms classical; much better robustness; fewer parameters; faster training. |
| CIFAR100  | ~32.25%             | ~41.69%          | ~9.44% increase; hybrid shows meaningful gains on more complex dataset. |
| STL10     | ~63.76%             | ~74.05%          | ~10.29% increase over classical. |

Other findings:  
- Hybrid models train **5–12× faster** in many settings.  
- Use **6–32% fewer parameters**.  
- Better adversarial robustness on simpler datasets; more comparable for complex ones.  
- Lower CPU & memory usage in many cases.  

---

## 🗂 Repository Structure

```

.
├── codes/                # Python scripts: training (hybrid & classical), evaluation, utilities
├── data/                 # Dataset files or scripts to download/preprocess MNIST, CIFAR100, STL10
├── figures/              # Graphs, plots, confusion matrices, feature visualizations, etc.
├── outputs/              # Text files: logs, metrics, raw output from code runs
├── report/               # LaTeX sources for the formal write-up (paper/report)
├── references.bib        # Bibliographic database for citations
├── LICENSE               # MIT License
└── README.md             # This file

````

---

## ⚙️ Requirements & Setup

Make sure you have the following:

- Python 3.8 or higher  
- Key Python libraries:  
  ```bash
  pip install torch torchvision pennylane matplotlib scikit-learn numpy
````

* Optional: GPU (for classical CNNs) to speed up training. Quantum circuits are simulated, so CPU is usable but slower.

---

## 🚀 Usage

Here’s how to use this repository:

1. **Clone the repo**

   ```bash
   git clone https://github.com/adnanphp/Hybrid-Quantum-Classical.git
   cd Hybrid-Quantum-Classical
   ```

2. **Prepare data**

   * Run preprocessing scripts in `data/` if any (or download datasets if not included).
   * Ensure folder structure matches what the training scripts expect (e.g., `data/MNIST`, etc.).

3. **Train models**

   * Classical model:

     ```bash
     python codes/train_classical.py --dataset MNIST --epochs 50
     ```
   * Hybrid model:

     ```bash
     python codes/train_hybrid.py --dataset MNIST --epochs 50
     ```

4. **Evaluate models**

   ```bash
   python codes/evaluate.py --model hybrid --dataset CIFAR100
   ```

5. **View outputs & figures**

   * Logs and numerical outputs are in `outputs/`
   * Plots (loss curves, accuracy curves, confusion matrices etc.) are in `figures/`

---

## 🧪 Experimental Setup

* **Epochs**: 50 per experiment
* **Datasets**: MNIST, CIFAR100, STL10
* **Adversarial robustness** tested under ε = 0.1 perturbations
* **Hybrid architecture**: parameterized quantum circuits + classical CNN components

---

## 🔮 Future Work

* Explore deployment on real quantum hardware (beyond simulation)
* Try deeper / more complex quantum circuit designs
* Apply to other data modalities (e.g., NLP, time series)
* Improve robustness especially for high-dimensional datasets

---

## 👤 Author

Muhammad Adnan Shahzad


---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
