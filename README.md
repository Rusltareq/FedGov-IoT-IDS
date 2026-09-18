# Blockchain-Governed Federated Learning for IoT Intrusion Detection

Reference implementation accompanying the PhD research and papers below, covering:

1. **Federated intrusion detection** on the CIC-ToN-IoT dataset under a *double non-IID* client split (protocol-based grouping + Dirichlet(α) class skew), comparing three aggregation algorithms — **FedAvg**, **FedProx**, and **FedDyn**.
2. **Active on-chain governance**: a Solidity smart contract (`ReputationManager`) that autonomously tracks each client's trust score and permanently isolates a client from future training rounds once its behavior crosses a policy threshold — with no human intervention required.

This repository contains the training/experiment code and the governance smart contract. It does **not** contain the raw dataset, any private keys, or any personally identifying credentials — see [Data](#data) and [Security notes](#security-notes) below.

## Related publications

- *FedTri-IDS* — submitted to *SN Computer Science*.
- *FedGov_XAI: Active On-Chain Governance and Trustworthy Explainable AI for Federated Intrusion Detection in IoT Networks* — submitted to *The Journal of Supercomputing* (Springer).
- *Client-Scale Generalization of Federated Aggregation in IoT Intrusion Detection: Accuracy, Stability, and Communication-Overhead Equivalence* — submitted to *Cluster Computing* (Springer).

If you use this code, please cite the relevant paper (see [Citation](#citation)).

## Repository structure

```
.
├── federated_ids_training.py           # FedAvg / FedProx / FedDyn training + evaluation framework
├── reputation_governance_simulation.py  # Replays per-client results against the on-chain governance contract
├── ReputationManager.sol                # The governance smart contract (Solidity ^0.8.19)
├── ReputationManager.abi.json           # ABI used by reputation_governance_simulation.py
├── requirements.txt
└── LICENSE
```

## Data

This repository does **not** include the CIC-ToN-IoT dataset. Obtain it from the official source (University of Queensland / UNSW Canberra Cyber, *"Towards a Standard Feature Set for Network Intrusion Detection System Datasets"*, Sarhan et al., 2022) and place the parquet file locally, e.g.:

```
data/CIC-ToN-IoT-V2.parquet
```

`federated_ids_training.py` expects the standard flow-based feature columns described in the code (`SCALING_FEATURES`, `OTHER_NUMERIC_FEATURES`, plus a `Protocol` and a `Label` column).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended but not required (the code falls back to CPU automatically). The original experiments (80 rounds × 3 algorithms × 3 seeds) were run on Google Colab with GPU acceleration; a full run on CPU will be considerably slower.

## Running the federated training experiment

```bash
python federated_ids_training.py \
    --data-path data/CIC-ToN-IoT-V2.parquet \
    --output-dir outputs/fl_outputs_v2 \
    --rounds 80 \
    --seeds 42 123 2024
```

This will:

1. Load and preprocess the dataset with leakage-free splitting (train/val/test split performed **before** any scaler or feature-selector is fit).
2. Partition the data across 10 simulated clients using the double non-IID scheme (protocol grouping + Dirichlet skew), printing a diagnostic table of each client's data distribution.
3. Train and evaluate FedAvg, FedProx, and FedDyn across all specified seeds.
4. Save per-seed personalization results, model checkpoints, a combined results CSV (`federated_results_all_seeds.csv`), paired significance tests (paired t-test and Wilcoxon signed-rank) between FedDyn and the strongest baseline, and convergence/fairness plots — all under `--output-dir`.

Key configurable parameters (`FLConfig` in the script, or via CLI flags) include the number of MI-selected features (`--top-k`), the Dirichlet concentration (`--dirichlet-alpha`), and the per-client training sample cap (`--max-client-train-samples`), which prevents any single large client from dominating weighted aggregation.

## Running the on-chain governance simulation

The governance mechanism is implemented as a Solidity smart contract and driven from Python via [web3.py](https://web3py.readthedocs.io/).

1. **Start a local test blockchain**, e.g. [Ganache](https://trufflesuite.com/ganache/) (`http://127.0.0.1:7545` by default).
2. **Deploy `ReputationManager.sol`** (e.g. via [Remix IDE](https://remix.ethereum.org/) connected to your local Ganache instance, or Hardhat/Truffle).
3. **Set the deployed contract address** as an environment variable:

   ```bash
   export CONTRACT_ADDRESS=0xYourDeployedContractAddress
   export RPC_URL=http://127.0.0.1:7545   # optional, this is the default
   ```

4. **Run the simulation**:

   ```bash
   pip install web3
   python reputation_governance_simulation.py
   ```

The script registers four example clients, then replays real per-client false-positive-rate (FPR) results — obtained from the federated training runs above — against the contract's `reportAlert()` function. Clients that repeatedly exceed the governance policy threshold (default 15% FPR, intentionally distinct from the 0.333 attack-decision threshold used for classification) are automatically isolated on-chain once their reputation score reaches zero.

`ReputationManager.abi.json` matches the exact ABI used in the original experiments. If you extend the contract (e.g. adding `reportAlertBatch`), regenerate the ABI from your compiler output and update `CONTRACT_ABI_PATH` accordingly.

## Security notes

- No API keys, private keys, or credentials are hardcoded anywhere in this repository.
- `VALIDATOR_PRIVATE_KEY` is read only from an environment variable and is intended for a **local test network account** (e.g. Ganache). Never use a mainnet or real-funds private key with this script, and never commit a `.env` file containing one.
- The dataset is not redistributed here; see [Data](#data) for how to obtain it from its official source.

## License

Released under the [MIT License](LICENSE) — you are free to use, modify, and redistribute this code, including for commercial purposes, provided the original copyright notice is retained.

## Citation

```bibtex
@article{khudhair_fedtri_ids,
  author  = {Khudhair, Rusul Tareq and Goh, Chin Hock and Abu Bakar, Asmidar},
  title   = {FedTri-IDS},
  journal = {SN Computer Science},
  note    = {Under review}
}

@article{khudhair_fedgov_xai,
  author  = {Khudhair, Rusul Tareq and Goh, Chin Hock and Abu Bakar, Asmidar},
  title   = {Active On-Chain Governance and Trustworthy Explainable AI for Federated Intrusion Detection in IoT Networks},
  journal = {The Journal of Supercomputing},
  note    = {Under review}
}

@article{khudhair_client_scale,
  author  = {Khudhair, Rusul Tareq and Goh, Chin Hock and Abu Bakar, Asmidar},
  title   = {Client-Scale Generalization of Federated Aggregation in IoT Intrusion Detection: Accuracy, Stability, and Communication-Overhead Equivalence},
  journal = {Cluster Computing},
  note    = {Under review}
}
```

## Author

Rusul Tareq Khudhair — PhD Candidate, Institute of Power Engineering, Universiti Tenaga Nasional (UNITEN), Malaysia.
