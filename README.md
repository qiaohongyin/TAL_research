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


## 2. System and Hardware Requirements

### 1. Tested Environment

The code has been tested in the following environment:

- **Operating System**: Ubuntu 20.04 / 22.04
- **Python**: 3.10
- **GPU**: NVIDIA GPU with CUDA support  
  - Tested on **NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM)**
- **CUDA**: 13.0
- **NVIDIA Driver**: 580.95.05
- **Deep Learning Frameworks**:
  - PyTorch 2.9.1
  - PyTorch Lightning 2.1.4

### 2. Environment Setup

| Package | Installation Command | Purpose |
| :--- | :--- | :--- |
| **Hydra** | `pip install hydra-core` | Configuration management. |
| **Lightning** | `pip install lightning` | Deep learning training framework. |
| **Utilities** | `pip install pandas scikit-learn` | Data handling and Macro-F1 score calculation. |



## 3.Reproducing the Experiments

This section describes how to reproduce the experimental results reported in the main paper.

---

### 1. Preparation

Please download all pretrained models from the following link:

https://drive.google.com/drive/folders/1H4ZQGA3_hw7AE6M4VmlLSZyvcep8RWbn

Then place the downloaded folders **inside the `TAL_research/` directory**.


### 2. Reproducing Table II: TAS Performance

#### 2.1 TAS (w/o DA)

Run the training script:
```bash
#train
python student.py mode=train --config-name dep_without_DA.yaml
#test
python student.py mode=test  --config-name dep_without_DA.yaml
```
---

#### 2.2 TAS (Ours)

Run the training script:
```bash
#train
python student.py mode=train --config-name dep_with_DA.yaml
#test
python student.py mode=test  --config-name dep_with_DA.yaml
```

### 3. Reproducing Table IV: Worker-Independent Results

Worker-Independent results are obtained by training the model on four users and averaging the results.

#### 3.1 TAS (w/o DA)

Run the training script:
```bash
#train--U0103
python student.py mode=train --config-name U0103_without_DA.yaml
#test--U0103
python student.py mode=test  --config-name U0103_without_DA.yaml

#train--U0105
python student.py mode=train --config-name U0105_without_DA.yaml
#test--U0105
python student.py mode=test  --config-name U0105_without_DA.yaml

#train--U0107
python student.py mode=train --config-name U0107_without_DA.yaml
#test--U0107
python student.py mode=test  --config-name U0107_without_DA.yaml

#train--U0108
python student.py mode=train --config-name U0108_without_DA.yaml
#test--U0108
python student.py mode=test  --config-name U0108_without_DA.yaml
```

---

#### 3.2 TAS (Ours)
Run the training script:
```bash
#train--U0103
python student.py mode=train --config-name U0103_with_DA.yaml
#test--U0103
python student.py mode=test  --config-name U0103_with_DA.yaml

#train--U0105
python student.py mode=train --config-name U0105_with_DA.yaml
#test--U0105
python student.py mode=test  --config-name U0105_with_DA.yaml

#train--U0107
python student.py mode=train --config-name U0107_with_DA.yaml
#test--U0107
python student.py mode=test  --config-name U0107_with_DA.yaml

#train--U0108
python student.py mode=train --config-name U0108_with_DA.yaml
#test--U0108
python student.py mode=test  --config-name U0108_with_DA.yaml
```

The final Worker-Independent result is the arithmetic mean of the four runs.

Although this requires some manual effort, we kindly ask you to record the results for each of the four users and manually compute their arithmetic mean to obtain the final Worker-Independent score. Thank you.

