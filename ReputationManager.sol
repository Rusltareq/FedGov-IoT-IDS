// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * ReputationManager
 * ------------------
 * Upgrade path from the previous "AlertLogger" contract (passive logging only)
 * toward the "Active On-Chain Governance" mechanism described in Phase 2 of
 * the research proposal.
 *
 * Note: reportAlert() is split into two small internal helpers
 * (_applyPenalty / _recordAlert) purely to keep the compiler's stack usage
 * low (avoids the "Stack too deep" error). The external interface (function
 * names, inputs, outputs) is unchanged, so any existing ABI/scripts built
 * against the previous version still work.
 */
contract ReputationManager {

    address public owner; // acts as the trusted PoA validator in this prototype

    uint256 public constant INITIAL_REPUTATION = 100;

    mapping(address => uint256) public reputation;
    mapping(address => bool) public isRegistered;
    mapping(address => bool) public isIsolated;

    struct Alert {
        address client;
        string algo;             // e.g. "FedAvg", "FedProx", "FedDyn"
        string dataset;          // e.g. "CIC-ToN-IoT"
        uint256 sampleIndex;
        uint256 scoreScaled;     // prediction score * 1e6 (Solidity has no floats)
        uint256 thresholdScaled; // decision threshold * 1e6
        uint256 penaltyApplied;  // reputation points deducted by this alert (0 if none)
        uint256 timestamp;
        bool triggeredPenalty;
    }

    Alert[] public alerts;

    /// @dev Bundles one alert's fields so reportAlertBatch() takes a single
    ///      calldata array instead of five parallel arrays (which was the
    ///      real cause of the "stack too deep" errors above).
    struct AlertInput {
        address client;
        uint256 sampleIndex;
        uint256 scoreScaled;
        uint256 thresholdScaled;
        uint256 penaltyAmount;
    }

    event ClientRegistered(address indexed client, uint256 initialReputation);
    event AlertLogged(uint256 indexed alertId, address indexed client, uint256 scoreScaled, uint256 timestamp);
    event ReputationUpdated(address indexed client, uint256 oldReputation, uint256 newReputation);
    event ClientIsolated(address indexed client, uint256 atAlertId, uint256 timestamp);

    modifier onlyOwner() {
        require(msg.sender == owner, "ReputationManager: caller is not the trusted validator");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    /// @notice Register a new federated client before it can be scored.
    function registerClient(address client) external onlyOwner {
        if (!isRegistered[client]) {
            isRegistered[client] = true;
            reputation[client] = INITIAL_REPUTATION;
            emit ClientRegistered(client, INITIAL_REPUTATION);
        }
    }

    /**
     * @notice Log an intrusion/anomaly alert tied to a client's update and,
     *         if the score crosses the given threshold, apply a reputation
     *         penalty. If reputation reaches zero, the client is isolated
     *         on-chain (future FL rounds should check isClientAllowed()).
     * @return isolatedNow true if this call is what pushed the client's
     *         reputation to zero.
     */
    function reportAlert(
        address client,
        string calldata algo,
        string calldata dataset,
        uint256 sampleIndex,
        uint256 scoreScaled,
        uint256 thresholdScaled,
        uint256 penaltyAmount
    ) external onlyOwner returns (bool isolatedNow) {
        require(isRegistered[client], "ReputationManager: client not registered");

        bool triggered = scoreScaled >= thresholdScaled;
        uint256 penaltyApplied;

        if (triggered) {
            (penaltyApplied, isolatedNow) = _applyPenalty(client, penaltyAmount);
        }

        uint256 alertId = _recordAlert(
            client, algo, dataset, sampleIndex, scoreScaled, thresholdScaled, penaltyApplied, triggered
        );

        if (isolatedNow) {
            emit ClientIsolated(client, alertId, block.timestamp);
        }
    }

    function _applyPenalty(address client, uint256 penaltyAmount)
        internal
        returns (uint256 penaltyApplied, bool isolatedNow)
    {
        if (isIsolated[client]) {
            return (0, false);
        }
        uint256 oldRep = reputation[client];
        uint256 newRep = oldRep > penaltyAmount ? oldRep - penaltyAmount : 0;
        penaltyApplied = oldRep - newRep;
        reputation[client] = newRep;
        emit ReputationUpdated(client, oldRep, newRep);
        if (newRep == 0) {
            isIsolated[client] = true;
            isolatedNow = true;
        }
    }

    function _recordAlert(
        address client,
        string calldata algo,
        string calldata dataset,
        uint256 sampleIndex,
        uint256 scoreScaled,
        uint256 thresholdScaled,
        uint256 penaltyApplied,
        bool triggered
    ) internal returns (uint256 alertId) {
        alerts.push(Alert({
            client: client,
            algo: algo,
            dataset: dataset,
            sampleIndex: sampleIndex,
            scoreScaled: scoreScaled,
            thresholdScaled: thresholdScaled,
            penaltyApplied: penaltyApplied,
            timestamp: block.timestamp,
            triggeredPenalty: triggered
        }));
        alertId = alerts.length - 1;
        emit AlertLogged(alertId, client, scoreScaled, block.timestamp);
    }

    /**
     * @notice Batched version of reportAlert(): records multiple alerts in a
     *         single transaction to amortise fixed per-transaction gas costs
     *         (base 21000 gas + calldata overhead) across many alerts.
     *         Reuses the same _applyPenalty / _recordAlert internal helpers,
     *         so per-alert governance logic is identical to the individual path.
     */
    function reportAlertBatch(
        AlertInput[] calldata inputs,
        string calldata algo,
        string calldata dataset
    ) external onlyOwner returns (uint256 isolatedCount) {
        for (uint256 i = 0; i < inputs.length; i++) {
            require(isRegistered[inputs[i].client], "ReputationManager: client not registered");

            bool triggered = inputs[i].scoreScaled >= inputs[i].thresholdScaled;
            uint256 penaltyApplied;
            bool isolatedNow;

            if (triggered) {
                (penaltyApplied, isolatedNow) = _applyPenalty(inputs[i].client, inputs[i].penaltyAmount);
            }

            uint256 alertId = _recordAlert(
                inputs[i].client, algo, dataset, inputs[i].sampleIndex,
                inputs[i].scoreScaled, inputs[i].thresholdScaled, penaltyApplied, triggered
            );

            if (isolatedNow) {
                emit ClientIsolated(inputs[i].client, alertId, block.timestamp);
                isolatedCount++;
            }
        }
    }

    /// @notice Meant to be called by the FL aggregator before using a client's update.
    function isClientAllowed(address client) external view returns (bool) {
        return isRegistered[client] && !isIsolated[client];
    }

    function getReputation(address client) external view returns (uint256) {
        return reputation[client];
    }

    function totalAlerts() external view returns (uint256) {
        return alerts.length;
    }

    function getAlert(uint256 index) external view returns (
        address client,
        string memory algo,
        string memory dataset,
        uint256 sampleIndex,
        uint256 scoreScaled,
        uint256 thresholdScaled,
        uint256 penaltyApplied,
        uint256 timestamp,
        bool triggeredPenalty
    ) {
        Alert memory a = alerts[index];
        return (
            a.client,
            a.algo,
            a.dataset,
            a.sampleIndex,
            a.scoreScaled,
            a.thresholdScaled,
            a.penaltyApplied,
            a.timestamp,
            a.triggeredPenalty
        );
    }
}
