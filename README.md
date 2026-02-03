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


1. Preparation
Ensure you have the environment set up and the necessary checkpoints (.ckpt files) placed in your designated storage path.

2. Reproducing Table II: TAS Performance
    TAS (w/o DA)
        1.Open configs/imu/deep-conv-lstm.yaml.
        2.Update assistant_ckpt_path to the path of work_dep_assistant_without_DA.ckpt.
        3.Comment out the session configuration line:
            # - [U0101, S0700]
        4.Run the training script:
            python student.py mode=train --config-name deep-conv-lstm.yaml
    TAS (Ours)
        1.Open configs/imu/deep-conv-lstm.yaml.
        2.Update assistant_ckpt_path to the path of work_dep_assistant_with_DA.ckpt.
        3.Ensure the session line is uncommented:
            - [U0101, S0700]
        4.Run the training script:
            python student.py mode=train --config-name deep-conv-lstm.yaml
3. Reproducing Table III (Ablation Study)
The ablation study results are obtained by manually modifying the loss function components in student.py.
L2: Feature Knowledge Distillation (feature_kd)
L3: Spatial Learning
L4: Temporal Transfer
        1.Open configs/imu/deep-conv-lstm.yaml.
        2.Update assistant_ckpt_path to the path of work_dep_assistant_without_DA.ckpt.
        3.Comment out the session configuration line:
            # - [U0101, S0700]
        4.Set the corresponding loss term(s) to zero.We recommend assigning zero to the composed term l (instead of only setting the raw loss to zero), so that both the weighted loss and its regularizer are removed from the total objective.
        5.Run the training script:
            python student.py mode=train --config-name deep-conv-lstm.yaml

4.Reproducing Table Ⅳ (Worker-Independent Results)
Worker-Independent results require running the model across four different users and averaging the final metrics.
For each user configuration file (deep-conv-lstm_U0103.yaml, U0105, U0107, U0108):
    1.Update the assistant_ckpt_path to the corresponding user checkpoint.
    2.Toggle DA:
        With DA: Keep - [U0101, S0700] active.
        Without DA: Comment out - [U0101, S0700].
    3.Run the training script for each user:
        # Example for U0103
        python student.py mode=train --config-name deep-conv-lstm_U0103.yaml
The final Worker-Independent score reported in the paper is the arithmetic mean of the results obtained from the four users mentioned above. Please calculate the average of these four experimental runs to verify the reported values.