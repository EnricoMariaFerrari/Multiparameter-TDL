# Image Multiparameter Filtration Learning for Enhancing Explainability in Neural Networks

This repository contains the implementation of a neural network framework (**TopoNet**) that integrates **topological descriptors** into image classification pipelines using **multiparameter filtrations** and **Euler characteristic profiles**.  
The goal is to improve **explainability** of neural networks through **topological data analysis (TDA)** while maintaining strong performance on medical imaging datasets.

---

## Project Structure
```text
project_root/
│
├── main.py # Training, validation, and testing script
│
├── models/
│ └── toponet.py # TopoNet definition
│
├── training/
│ ├── train.py # train, test, train_model functions
│ ├── loss.py # Contrastive & classification loss
│ └── optimizer.py # Optimizer configuration
│
├── utils/
│ ├── data.py # Dataloaders and preprocessing
│ ├── metrics.py # AUC and other metrics
│ ├── label_map.py # Dataset-specific label mappings
│ └── formatter.py # Format signed measures
│
├── explainability/
│ ├── surfaces.py # Euler surfaces visualization
│ ├── outputs.py # Model output computations
│ └── plots.py # Plotting critical points, etc.
│
├── results/ # Saved models and explainability outputs
│ └── .gitignore
│
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── .gitignore
```
---

## Getting Started

1. Clone the Repository
git clone https://github.com/<your-username>/TopoNet-Multiparameter-Filtrations.git
cd TopoNet-Multiparameter-Filtrations

2. Create a Virtual Environment (Optional but Recommended)
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Install Requirements
pip install -r requirements.txt

4. Run Training
To start training with default arguments:
python main.py

To change dataset, batch size, number of epochs, etc.:
python main.py --data_name PathMNIST --epochs 10 --batch_size 64

## 📊 Datasets
This project uses datasets from MedMNIST:
- PathMNIST
- OCTMNIST
- TissueMNIST
They are automatically downloaded the first time you run the code.

## 🔎 Explainability
To generate topological explanations (Euler characteristic surfaces, critical points, etc.):
python main_explainability.py
(More details coming soon.)

## 🛠 Requirements
- Python >= 3.8
- PyTorch
- multipers (for topological signatures)
- medmnist
- scikit-learn, numpy, matplotlib, joblib, etc.

Install everything with:
pip install -r requirements.txt

## 📂 Results
Trained models and explainability outputs are saved in the results/ folder.

## 📄 License
MIT License (optional — remove if not needed).

## ✍️ Author
Created by EnricoMariaFerrari

## 🌐 Acknowledgements
- medmnist
- multipers
