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


Reproducing the Experiments

This section describes how to reproduce the experimental results reported in the main paper.

1. Preparation

Before running any experiments, please ensure that:

The environment is correctly set up according to the instructions in this repository.

All required pretrained checkpoints (.ckpt files) are downloaded and placed in your designated storage path.

The checkpoint paths are correctly specified in the corresponding configuration files.

2. Reproducing Table II: TAS Performance
2.1 TAS (w/o DA)

Open the configuration file:

configs/imu/deep-conv-lstm.yaml


Update assistant_ckpt_path to point to:

work_dep_assistant_without_DA.ckpt


Comment out the following session configuration line:

# - [U0101, S0700]


Run the training script:

python student.py mode=train --config-name deep-conv-lstm.yaml

2.2 TAS (Ours)

Open the configuration file:

configs/imu/deep-conv-lstm.yaml


Update assistant_ckpt_path to point to:

work_dep_assistant_with_DA.ckpt


Ensure the following session configuration line is uncommented:

- [U0101, S0700]


Run the training script:

python student.py mode=train --config-name deep-conv-lstm.yaml

3. Reproducing Table III: Ablation Study

The ablation study is conducted by manually disabling specific loss components in student.py.

The evaluated components are:

L2: Feature Knowledge Distillation (feature_kd)

L3: Spatial Learning

L4: Temporal Transfer

Steps:

Open the configuration file:

configs/imu/deep-conv-lstm.yaml


Update assistant_ckpt_path to point to:

work_dep_assistant_without_DA.ckpt


Comment out the following session configuration line:

# - [U0101, S0700]


Set the corresponding loss term(s) to zero in student.py.

Note:
We recommend assigning zero to the composed loss term ℓ (rather than only setting the raw loss to zero), so that both the weighted loss and its associated regularizer are fully removed from the total objective.

Run the training script:

python student.py mode=train --config-name deep-conv-lstm.yaml

4. Reproducing Table IV: Worker-Independent Results

Worker-Independent results are obtained by training the model on four different users and averaging the final performance.

The corresponding configuration files are:

deep-conv-lstm_U0103.yaml

deep-conv-lstm_U0105.yaml

deep-conv-lstm_U0107.yaml

deep-conv-lstm_U0108.yaml

For each user:

Update assistant_ckpt_path to the corresponding user-specific assistant checkpoint.

Toggle Domain Adaptation (DA):

With DA: Keep the following line active:

- [U0101, S0700]


Without DA: Comment out the line above.

Run the training script.
Example for user U0103:

python student.py mode=train --config-name deep-conv-lstm_U0103.yaml


The final Worker-Independent score reported in the paper is computed as the arithmetic mean of the results obtained from these four users.
Please average the four experimental results to verify the reported values.