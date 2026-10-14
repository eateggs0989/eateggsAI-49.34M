# eateggsAI-49.343M v0.1

eateggsAI-49.343M is an experimental SLM created as a low-VRAM challenge to train a 49.343M parameter transformer on a GTX 1060 4GB GPU using PyTorch.

---

# Project Overview

This project focuses on building and training a lightweight Small Language Model (SLM) completely from scratch using PyTorch.

The main goal of this experiment was to explore whether older consumer GPUs with limited VRAM can still train transformer-based language models efficiently.

Instead of using expensivecloud GPUs (like A100s or H100s), this project was trained on a GTX 1060 4GB GPU as part of a low-VRAM AI challenge.

---
## Why This Project Matters

Modern LLM/SLM development often depends on expensive enterprise GPUs with massive amounts of VRAM.

This project explores a different question: **Can custom transformer-based language models still be trained efficiently on older, consumer-grade hardware?**

`eateggsAI-49.343M` was built as an experimental low-VRAM AI challenge to push the absolute limits of a **GTX 1060 4GB GPU**—proving that you can achieve stable transformer training and meaningful text generation from scratch without needing cloud clusters or expensive AI hardware.

---

### ⚙️ The Low-VRAM Engineering Breakdown

To squeeze a ~49M parameter model onto a 4GB card without hitting out-of-memory (OOM) errors, the pipeline relies on several critical optimizations:

* **Automatic Mixed Precision (AMP):** Utilizes PyTorch's `autocast` and `GradScaler` to run critical matrix math in FP16, effectively cutting activation memory footprints in half.
* **Controlled Sequence Length:** Capped the attention block size at **256 tokens** to prevent the quadratic memory explosion ($O(T^2)$) common in longer transformer contexts.
* **Optimized Micro-Batching:** Configured a lean batch size of **11** to ensure peak gradient memory stays comfortably below the 4GB ceiling.
* **Balanced Architecture Design:** Features a custom 6-layer structure with an embedding dimension of 384, balancing model capacity with strict hardware reality.

  
# Tested Environment

| Component | Version |
|---|---|
| Python | 3.11 |
| PyTorch | Latest |
| CUDA | Supported |

# Hardware Used

| Component | Specification |
|---|---|
| Laptop | Acer i5 |
| GPU | NVIDIA GTX 1060 |
| VRAM | 4GB |

---

# Model Configuration

| Setting | Value |
|---|---|
| Parameters | 49,343,232 (~49.34M) |
| Transformer Layers | 6 |
| Attention Heads | 8 |
| Embedding Size | 384 |
| Context Length | 256 |
| Vocabulary Size | 50,257 |
| Dropout | 0.1 |
| Architecture | GPT-style Decoder-Only Transformer |

---

# Training Results

| Metric | Value |
|---|---|
| Training Time | ~12 Hours |
| VRAM Usage | ~2.7GB |
| Final Loss | 3.2 – 3.3 |
| Framework | PyTorch |
| Precision | Mixed Precision (AMP) |

---
# Training Stack

## Dataset Pipeline
- Multi-domain dataset streaming
- Dataset balancing
- Data cleaning
- Deduplication
- GPT2 tokenization
- Fixed-length block packing
- PyTorch tensor conversion

# Dataset Sources

The training dataset was built using multiple domains:

- Wikipedia
- AG News
- SQuAD
- PG19 Books

<img width="536" height="233" alt="Screenshot 2026-05-18 170312" src="https://github.com/user-attachments/assets/c83d65b3-88ef-40c4-aebf-5cf42c572f7e" />

# Download RawDataSets

<img width="1055" height="497" alt="Screenshot 2026-05-18 171040" src="https://github.com/user-attachments/assets/8389f0f8-f378-4592-9796-d99235b84392" />

The dataset was cleaned and tokenized before training.

## Model
- GPT-style decoder-only transformer architecture built from scratch
- Multi-head causal self-attention
- GELU activation
- Residual connections
- Layer normalization
- Learned positional embeddings
- Causal attention masking

## Training
- AdamW optimizer
- Mixed precision training (AMP)
- Gradient scaling
- Cross-entropy loss
- CUDA acceleration
- Low-VRAM optimization
- Model weight saving
- 
# Features
- Built-from-scratch Transformer Architecture (~49.3M parameters)
- Standard Hugging Face GPT-2 Tokenizer (`vocab_size = 50,257`)
- Fully implemented in PyTorch
- Multi-Head Self Attention
- Mixed Precision Training
- Low VRAM Optimization
- Multi-domain Dataset Pipeline
- Fully implemented in PyTorch

---

# Project Pipeline

Datasets
↓
Data Cleaning
↓
Tokenization (GPT2 Tokenizer)
↓
Block Creation
↓
Transformer Training
↓
Loss Optimization
↓
Model Saving

---
# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/eateggsAI-30M.git
cd eateggsAI-30M
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset Preparation

Download datasets:

```bash
python dataset/download_datasets.py
```

Clean dataset:

```bash
python dataCleaner.py
```

Prepare token blocks:

```bash
python dataset/prepare_blocks.py
```
# How To Train

Run the training script:

```bash
python training/train.py
```


The trained model weights will be saved as:

```text
gpt_6layer.pt
```

---

# How To Run Inference

Run the inference script:

```bash
python inference/generate.py
```

Example prompt:

```text
Why is the sky blue?
```

# Sample Generations

Below are real outputs generated by eateggsAI-49.34M during inference testing.

<img width="1234" height="815" alt="Screenshot 2026-05-20 031607" src="https://github.com/user-attachments/assets/c04ca551-3679-4ac9-856b-925b662e43ab" />


# Purpose Of This Project

This project was created to:

- Learn transformer architectures deeply
- Understand GPT training internally
- Explore low-VRAM AI systems
- Build a fully custom LLM pipeline
- Prove older GPUs can still train language models

---

# Technologies Used

- Python
- PyTorch
- CUDA
- Hugging Face Datasets
- Transformers
- GPT2Tokenizer

---

# What This Project Demonstrates

eateggsAI-~49.34M demonstrates:

- Training a Small Language Model SLM on low-VRAM consumer hardware
- Building transformer architectures from scratch
- Implementing causal self-attention manually
- Creating custom dataset pipelines
- GPT2 tokenization workflows
- Multi-domain language model training
- Mixed precision optimization using AMP
- Efficient transformer experimentation on older GPUs

This project can be used as:
- An educational transformer implementation
- A beginner-friendly GPT architecture reference
- A low-VRAM LLM training experiment
- A PyTorch NLP learning project
- A foundation for future fine-tuning experiments

---

# Status

Current Version:
```text
eateggsAI-49.34M v0.1
