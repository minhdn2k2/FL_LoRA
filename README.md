# FL_LoRA

This repository contains implementations of several **Federated Learning (FL) algorithms for LoRA-based fine-tuning**.

## 🚀 Implemented Algorithms

The following algorithms are implemented:

- **FFA**: *Improving LoRA in Privacy-preserving Federated Learning*
- **FedIT**: *FL + LoRA (native)*
- **LoRA-FAIR**: *LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement*
- **FedEx-LoRA**: *FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Foundation Models*
- **RoLoRA**: *Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA*

---

## 📊 Training Setup

We conduct experiments on **GLUE benchmark tasks**:

- `MNLI`
- `QQP`
- `SST2`
- `QNLI`
- `RTE`

### Hyperparameters

We evaluate under different federated settings:

- **Number of clients**:
  - `10`
  - `50` (with `selected_ratio = 0.1`)

- **Data heterogeneity (Dirichlet split)**:
  - `split_coef = 0.1` (highly non-IID)
  - `split_coef = 0.5` (moderately non-IID)

- **Random seed**:
  - `seed = 123`

---

## 🏃 How to Train

All experiments are organized into `.sh` scripts.

### Example: Run FedIT with Dirichlet non-IID coefficient 0.1 and 10 clients

```bash
chmod +x run/run_fedit_seed123_dir01_cli10.sh
./run/run_fedit_seed123_dir01_cli10.sh