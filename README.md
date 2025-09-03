# 🧠 Image Multiparameter Filtration Learning for Enhancing Explainability in Neural Networks

This repository implements **MultiTopoNet**, a topological convolutional neural network that integrates multiparameter filtration learning —via  Euler surface— into image classification pipelines.

The goal is to improve the explainability of neural networks through the bifiltration learning process, while maintaining high classification performance on medical imaging datasets.

---

## 📁 Project Structure

- [`main.py`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/blob/main/main.py)  
  Main training, validation, and testing script.

- [`models/`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/tree/main/models)  
  - [`toponet.py`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/blob/main/models/toponet.py): Definition of the TopoNet architecture.

- [`training/`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/tree/main/training)  
  - [`train.py`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/blob/main/training/train.py): Training, testing, and model orchestration.  
  - [`loss.py`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/blob/main/training/loss.py): Contrastive and classification loss functions.  
  - [`optimizer.py`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/blob/main/training/optimizer.py): Optimizer setup.

- [`utils/`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/tree/main/utils)  
  - [`data.py`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/blob/main/utils/data.py): Dataset loading and preprocessing.  
  - [`metrics.py`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/blob/main/utils/metrics.py): Evaluation metrics (AUC, etc.).  
  - [`label_map.py`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/blob/main/utils/label_map.py): Dataset-specific label mappings.  
  - [`formatter.py`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/blob/main/utils/formatter.py): Signed measure formatting utilities.

- [`explainability/`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/tree/main/explainability)  
  - [`surfaces.py`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/blob/main/explainability/surfaces.py): Euler surfaces computation and visualization.  
  - [`outputs.py`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/blob/main/explainability/outputs.py): Model outputs for explanation.  
  - [`plots.py`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/blob/main/explainability/plots.py): Plotting critical points and diagrams.

- [`results/`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/tree/main/results)  
  Folder for saved models and explainability outputs.

- [`requirements.txt`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/blob/main/requirements.txt)  
  List of Python dependencies.

- [`README.md`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/blob/main/README.md)  
  You are here 📄

- [`.gitignore`](https://github.com/EnricoMariaFerrari/Multiparameter-TDL/blob/main/.gitignore)  
  Files and folders excluded from version control.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/EnricoMariaFerrari/Multiparameter-TDL.git
cd Multiparameter-TDL
```

### 2. (Optional) Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run training

To start training with default parameters:

```bash
python main.py
```

Customize training:

```bash
python main.py --data_name PathMNIST --epochs 10 --batch_size 64
```

---

## 🧬 Datasets

This project uses datasets from [**MedMNIST**](https://medmnist.com):

- `PathMNIST`
- `OCTMNIST`
- `TissueMNIST`

They will be **automatically downloaded** the first time you run the code.

---

## 🔍 Explainability

To generate **topological explanations** (Euler characteristic surfaces, critical points, etc.), run:

```bash
python main_explainability.py --net_path path/to/trained_model.pth
```

> ℹ️ More configuration options and documentation coming soon.

---

## 🛠 Requirements

This project requires:

- Python ≥ 3.8  
- PyTorch (tested with 2.6.0)
- torchvision
- [multipers](https://github.com/simongresta/multipers) ≥ 2.3.1
- [medmnist](https://github.com/MedMNIST/MedMNIST)
- numpy
- matplotlib
- scikit-learn
- tqdm
- joblib

Install all dependencies with:

```bash
pip install -r requirements.txt
```

> 💡 For GPU users: you may want to install the CUDA-specific version of PyTorch manually.  
> Example:
> ```bash
> pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
> ```

---

## 📂 Results

- Trained models and explanation outputs are saved in the `results/` folder.
- Default subfolders are automatically created based on dataset and classes.

---

## 📄 License

MIT License  
*(Update or remove this section if a different license is used)*

---

## ✍️ Author

Developed by **Enrico Maria Ferrari**  
Feel free to reach out for questions or collaborations.

---

## 🙏 Acknowledgements

- [**MedMNIST**](https://github.com/MedMNIST/MedMNIST): Lightweight benchmark for medical image analysis.  
- [**multipers**](https://github.com/simongresta/multipers): Multiparameter persistent homology framework.
