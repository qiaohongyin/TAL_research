# Artifacts : A Teaching Assistant for Teacher–Student Learning

This repository accompanies the paper: **“A Teaching Assistant for Teacher–Student Learning: Knowledge Transfer from Skeleton to Inertial Sensing for Activity Recognition in Industrial Domains.”**

It provides the implementation, configuration files, and preprocessed data required to reproduce the Teacher–Assistant–Student (TAS) framework for transferring knowledge from skeleton-based models (ST-GCN) to IMU-based models (DeepConvLSTM).

## 1. Project Structure

The repository is organized as follows:
- `data/`: Preprocessed skeleton and IMU windows from the OpenPack dataset.
- `configs/`: Hydra-based YAML configuration files for experiments.
- `openpack_torch/`: Core library containing models, data modules, and evaluation utilities.
- `assistant.py`: Script to train the Assistant model (dual-wrist IMU).
- `student.py`: Script to train the Student model (single-wrist IMU) using the TAS framework.
- `README.md`: This guide.

## 2. Environment Setup

The code has been tested on **Python 3.10**. To ensure compatibility (especially with Hydra on Python 3.10), please follow these installation steps:

| Package | Installation Command | Purpose |
| :--- | :--- | :--- |
| **Hydra** | `pip install hydra-core --upgrade` | Configuration management (resolves Python 3.10 compatibility issues). |
| **Lightning** | `pip install lightning --upgrade` | Deep learning training framework. |
| **Utilities** | `pip install pandas scikit-learn` | Data handling and Macro-F1 score calculation. |

### Step 1: Train the Assistant Model
Run the student script using the best checkpoint from the teacher:
python assistant.py mode = train

### Step 2: Train the Student Model (TAS Framework)
Run the student script using the best checkpoint from the assistant:
python student.py mode = train