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




## 3.Reproducing the Experiments

This section describes how to reproduce the experimental results reported in the main paper.

---

### 1. Preparation

Ensure that the environment is correctly set up and Please download all of the pretrained models from the following link:

https://drive.google.com/drive/folders/1FO4r3D4Zd_HLPG5-8LtBs9VQlqe6YC37?dmr=1&ec=wgc-drive-hero-goto

And update the following paths in the configuration file to match your local environment:
   - Update `logdir` to a local directory for saving logs and checkpoints, for example:
     ```
     logdir: ./logs/student/
     ```

   - Update the dataset path to your local OpenPack data location:
     ```yaml
     dataset:
       path: /path/to/your/local/openpack_data/${user.name}/window/${session}.npz
     ```

---


### 2. Reproducing Table II: TAS Performance

#### 2.1 TAS (w/o DA)

1. Open the configuration file:
   configs/deep-conv-lstm.yaml

2. Update `assistant_ckpt_path` to the path of pretrained assistant checkpoint:
     ```
     work_dep_assistant_without_DA.ckpt
     ```


3. Comment out the following session configuration line:
   ```yaml
    # - [U0101, S0700]

4. Run the training script:
   python student.py mode=train --config-name deep-conv-lstm.yaml

---

#### 2.2 TAS (Ours)

1. Open the configuration file:
   configs/deep-conv-lstm.yaml

2. Update `assistant_ckpt_path` to the path of pretrained assistant checkpoint:
   ```
   work_dep_assistant_with_DA.ckpt
   ```
   
3. Ensure the following session configuration line is active:
    ```yaml
    - [U0101, S0700]

4. Run the training script:
   python student.py mode=train --config-name deep-conv-lstm.yaml

---

### 3. Reproducing Table III: Ablation Study

The ablation study is conducted by manually disabling loss components in student.py.

L2: Feature Knowledge Distillation (feature_kd)  
L3: Spatial Learning  
L4: Temporal Transfer  

Steps:

1. Open the configuration file:
   configs/deep-conv-lstm.yaml

2. Update `assistant_ckpt_path` to:
   work_dep_assistant_without_DA.ckpt

3. Comment out the following session configuration line:
   ```yaml
    # - [U0101, S0700]


4. Set the corresponding loss term(s) to zero.
   We recommend setting the composed loss term ℓ to zero so that both the weighted loss
   and its regularizer are removed from the total objective.

5. Run the training script:
   python student.py mode=train --config-name deep-conv-lstm.yaml

---

### 4. Reproducing Table IV: Worker-Independent Results

Worker-Independent results are obtained by training the model on four users and averaging the results.

Configuration files:
- deep-conv-lstm_U0103.yaml
- deep-conv-lstm_U0105.yaml
- deep-conv-lstm_U0107.yaml
- deep-conv-lstm_U0108.yaml

For each user:

1. Update `assistant_ckpt_path` to the corresponding user checkpoint.

2. Toggle DA:
   With DA:   keep   - [U0101, S0700]
   Without DA: comment out the line above.

3. Run training, e.g. for U0103:
   python student.py mode=train --config-name deep-conv-lstm_U0103.yaml

The final Worker-Independent result is the arithmetic mean of the four runs.

Although this requires some manual effort, we kindly ask you to record the results for each of the four users and manually compute their arithmetic mean to obtain the final Worker-Independent score. Thank you.

