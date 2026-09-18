"""
On-Chain Active Governance Simulation — ReputationManager
===========================================================
This script replays real per-client evaluation results (false-positive
rate, FPR) from the federated training runs (see
`federated_ids_training.py`) against the deployed `ReputationManager`
Solidity contract (see `contracts/ReputationManager.sol`).

It demonstrates the "active governance" mechanism described in the
accompanying papers: each client's FPR is compared, round by round,
against a policy threshold (default 0.15 — chosen deliberately
different from the 0.333 attack-decision threshold used elsewhere in
the pipeline). A client that repeatedly exceeds the threshold loses
reputation points and is automatically, permanently isolated on-chain
once its reputation reaches zero — no human intervention required.

Prerequisites
-------------
1. A running Ethereum test network reachable at RPC_URL (this project
   used Ganache locally: `http://127.0.0.1:7545`). Never point this at
   a mainnet or any network holding real funds.
2. The `ReputationManager` contract (contracts/ReputationManager.sol)
   already compiled and deployed to that network (e.g. via Remix IDE
   or Hardhat/Truffle), with its address and ABI available.
3. `pip install web3`

Configuration (all via environment variables — nothing sensitive is
hardcoded in this file):
    RPC_URL              Ethereum JSON-RPC endpoint (default: local Ganache)
    CONTRACT_ADDRESS      Deployed ReputationManager address (required)
    CONTRACT_ABI_PATH     Path to a JSON file containing the contract ABI
                           (default: contracts/ReputationManager.abi.json)
    VALIDATOR_PRIVATE_KEY Optional. Only needed if the validator account
                           is not unlocked on the node itself. Never commit
                           a real private key — use an environment variable
                           or a local .env file that is gitignored.

Usage
-----
    export CONTRACT_ADDRESS=0xYourDeployedContractAddress
    python reputation_governance_simulation.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from web3 import Web3

# ---------------------------------------------------------------------------
# Configuration — read from environment, nothing sensitive hardcoded here
# ---------------------------------------------------------------------------
RPC_URL = os.environ.get("RPC_URL", "http://127.0.0.1:7545")
CONTRACT_ADDRESS = os.environ.get("CONTRACT_ADDRESS")
CONTRACT_ABI_PATH = os.environ.get(
    "CONTRACT_ABI_PATH",
    str(Path(__file__).resolve().parent / "ReputationManager.abi.json"),
)
VALIDATOR_PRIVATE_KEY = os.environ.get("VALIDATOR_PRIVATE_KEY")  # optional, local test key only

GOVERNANCE_FPR_THRESHOLD = 0.15  # policy ceiling — distinct from the 0.333 attack-decision threshold
PENALTY_PER_VIOLATION = 20
SCALE = 1_000_000  # Solidity has no floats; scores/thresholds are scaled integers

# ---------------------------------------------------------------------------
# Example real per-client FPR data (four representative clients out of the
# original ten in the FedTri-IDS design), taken directly from this
# project's own personalized_{fedavg,fedprox,feddyn}_seed{42,123,2024}.csv
# evaluation outputs. Replace with your own results to reproduce this
# experiment on a different run.
#   client_1: clean client            (FPR consistently ~4-5%)
#   client_2: borderline/unstable     (FPR ~9-15%, exceeds threshold twice of eight runs)
#   client_5: persistent violator     (FPR ~18-22%, exceeds threshold every run)
#   client_6: worst-behaved client    (FPR ~29-30%, exceeds threshold every run)
# Format: (algorithm, seed, fpr)
# ---------------------------------------------------------------------------
REAL_FPR_DATA = {
    "client_1": [
        ("FedAvg", 42, 0.053609), ("FedAvg", 123, 0.050838), ("FedAvg", 2024, 0.044834),
        ("FedProx", 42, 0.053254), ("FedProx", 123, 0.045936), ("FedProx", 2024, 0.045971),
        ("FedDyn", 42, 0.041708), ("FedDyn", 123, 0.046256),
    ],
    "client_2": [
        ("FedAvg", 42, 0.135678), ("FedAvg", 123, 0.151759), ("FedAvg", 2024, 0.147739),
        ("FedProx", 42, 0.094472), ("FedProx", 123, 0.139698), ("FedProx", 2024, 0.150754),
        ("FedDyn", 42, 0.135678), ("FedDyn", 123, 0.115578),
    ],
    "client_5": [
        ("FedAvg", 42, 0.218075), ("FedAvg", 123, 0.220039), ("FedAvg", 2024, 0.194499),
        ("FedProx", 42, 0.214145), ("FedProx", 123, 0.201375), ("FedProx", 2024, 0.191552),
        ("FedDyn", 42, 0.201375), ("FedDyn", 123, 0.183694),
    ],
    "client_6": [
        ("FedAvg", 42, 0.292969), ("FedAvg", 123, 0.300781), ("FedAvg", 2024, 0.300781),
        ("FedProx", 42, 0.300781), ("FedProx", 123, 0.300781), ("FedProx", 2024, 0.300781),
        ("FedDyn", 42, 0.300781), ("FedDyn", 123, 0.300781),
    ],
}


def scale(x: float) -> int:
    return int(round(x * SCALE))


def load_abi(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_sender(w3: Web3, validator_address: str, private_key: str | None):
    """Returns a function that sends a contract transaction and waits for its receipt.
    Uses the node's own unlocked account if no private key is given (fine for a
    local Ganache/dev node); otherwise signs and sends a raw transaction."""

    def send_tx(fn):
        if not private_key:
            tx_hash = fn.transact({"from": validator_address, "gas": 3_000_000})
            return w3.eth.wait_for_transaction_receipt(tx_hash)
        nonce = w3.eth.get_transaction_count(validator_address)
        txn = fn.build_transaction({
            "from": validator_address, "nonce": nonce,
            "gas": 3_000_000, "gasPrice": w3.to_wei("1", "gwei"),
        })
        signed = w3.eth.account.sign_transaction(txn, private_key=private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        return w3.eth.wait_for_transaction_receipt(tx_hash)

    return send_tx


def main():
    if not CONTRACT_ADDRESS:
        raise SystemExit(
            "Set the CONTRACT_ADDRESS environment variable to your deployed "
            "ReputationManager address before running this script."
        )

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    print("Connected:", w3.is_connected(), "| RPC:", RPC_URL)

    abi = load_abi(CONTRACT_ABI_PATH)
    contract = w3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=abi)

    validator_address = w3.eth.accounts[0]
    # Use four fresh, unused local test accounts to represent the four example clients.
    # On a local Ganache instance these are throwaway test addresses with no real funds.
    client_addresses = {
        "client_1": w3.eth.accounts[3],
        "client_2": w3.eth.accounts[4],
        "client_5": w3.eth.accounts[5],
        "client_6": w3.eth.accounts[6],
    }

    send_tx = make_sender(w3, validator_address, VALIDATOR_PRIVATE_KEY)

    print("\n== Registering clients ==")
    for name, addr in client_addresses.items():
        send_tx(contract.functions.registerClient(addr))
        print(f"Registered {name} ({addr}) | reputation = {contract.functions.getReputation(addr).call()}")

    def report_alert(client_addr, algo, dataset, seed, fpr, penalty=PENALTY_PER_VIOLATION):
        """`seed` is passed in place of sampleIndex because results here are
        aggregated per experiment run, not per individual network sample."""
        send_tx(contract.functions.reportAlert(
            client_addr, algo, dataset, seed, scale(fpr), scale(GOVERNANCE_FPR_THRESHOLD), penalty
        ))
        rep = contract.functions.getReputation(client_addr).call()
        allowed = contract.functions.isClientAllowed(client_addr).call()
        status = "ISOLATED" if not allowed else "allowed"
        flag = "VIOLATION" if fpr >= GOVERNANCE_FPR_THRESHOLD else "ok"
        print(f"[{dataset}/{algo:8s} seed={seed:5d}] FPR={fpr:.3f} [{flag:9s}] "
              f"-> reputation = {rep:3d} | {status}")

    print("\n== Replaying real per-client FPR results against the governance contract ==")
    for client_name, addr in client_addresses.items():
        print(f"\n----- {client_name} ({addr[:10]}...) -----")
        for algo, seed, fpr in REAL_FPR_DATA[client_name]:
            report_alert(addr, algo, "CIC-ToN-IoT", seed, fpr)

    print("\n== Final state ==")
    for client_name, addr in client_addresses.items():
        rep = contract.functions.getReputation(addr).call()
        allowed = contract.functions.isClientAllowed(addr).call()
        print(f"{client_name}: reputation={rep:3d}, allowed_in_next_round={allowed}")

    print("\nTotal alerts recorded on-chain:", contract.functions.totalAlerts().call())


if __name__ == "__main__":
    main()
