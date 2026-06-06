# FedANC: Federated Learning with Adversarial Neural Cryptography

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

FedANC is a privacy‑preserving communication framework that combines **Federated Learning** and **Adversarial Neural Cryptography**. Multiple clients collaboratively train a global encryption‑decryption model **without sharing their local plaintexts or secret keys**. Each client runs a local Alice‑Bob‑Eve adversarial training, and only model parameters (Alice & Bob) are uploaded to the central server for FedAvg aggregation. This ensures that raw data never leaves the local device.

The project is built upon the original ANC paper (Abadi & Andersen, ICLR 2017) and extends it to a distributed, privacy‑friendly setting.
## Architecture

The following diagram shows the federated learning workflow:

        ┌─────────────┐
        │   Server    │
        │  (FedAvg)   │
        └──────┬──────┘
               │
     ┌─────────┼─────────┐
     │         │         │
     ▼         ▼         ▼
┌─────────┐┌─────────┐┌─────────┐
│Client 1 ││Client 2 ││Client 3 │
│ (P,K)   ││ (P,K)   ││ (P,K)   │
│ANC      ││ANC      ││ANC      │
└─────────┘└─────────┘└─────────┘

- Clients generate random plaintext `P` and key `K` locally.
- Each client performs local Alice‑Bob‑Eve adversarial training.
- Clients upload only the model parameters (Alice & Bob) to the server.
- The server aggregates parameters using FedAvg and distributes the updated global model.
## Key Features

- **Privacy by design** – Clients never share plaintext or keys; only model parameters are exchanged.
- **Adversarial training** – Local Alice‑Bob‑Eve games learn secure encryption without predefined algorithms.
- **Federated averaging** – Server aggregates parameters via FedAvg, reducing communication overhead.
- **Comprehensive evaluation** – Includes both centralized training (baseline) and federated learning.
- **Easy to run** – Simple command‑line interface, no external datasets required (random data generated on the fly).

## Results (on 16‑bit blocks)

| Method               | Decoder MSE | Eve MSE | Privacy |
|----------------------|-------------|---------|---------|
| Centralized          | 0.00263     | 0.945   | None    |
| Federated (FedANC)   | 0.06668     | 1.305   | Yes     |

- Decoder MSE = reconstruction error (lower is better)
- Eve MSE = adversary error (higher indicates better secrecy, 1.0 ≈ random guess)

## Requirements

- Python 3.8 – 3.12
- PyTorch 2.0+
- NumPy, Matplotlib, tqdm

## Installation

Clone the repository and install dependencies:

    git clone https://github.com/324bfy/FedANC.git
    cd FedANC
    pip install torch numpy matplotlib tqdm

(Optional) Create a virtual environment:

    python -m venv .venv
    source .venv/bin/activate       # Linux/macOS
    .venv\Scripts\activate          # Windows

## Usage

All experiments are launched via `run.py`.

- Centralized Training (baseline):  
  `python run.py train`

- Federated Learning:  
  `python run.py fed`

- Encryption / Decryption Demo:  
  `python run.py encrypt`

## File Structure

    .
    ├── run.py
    ├── settings.py
    ├── server.py
    ├── core/
    │   ├── crypto_nets.py
    │   ├── data_feeder.py
    │   ├── local_worker.py
    │   ├── standalone.py
    │   ├── utils.py
    │   └── evaluator.py
    └── models/

## Acknowledgements

This project is based on the [CryptoGAN](https://github.com/zrthxn/CryptoGAN) implementation (MIT License) of the paper "Learning to Protect Communications with Adversarial Neural Cryptography" by Abadi & Andersen. We have refactored the codebase, added federated learning extensions, and changed naming conventions.