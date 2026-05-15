from __future__ import annotations
from typing import Dict


def simulate_transfer(payload_kb: float, bandwidth_kbps: int, network_delay_ms: int) -> Dict[str, float]:
    bandwidth_kBps = max(bandwidth_kbps / 8.0, 1e-6)
    tx_ms = (payload_kb / bandwidth_kBps) * 1000.0
    transfer_ms = tx_ms + float(network_delay_ms)
    return {
        "payload_kb": float(payload_kb),
        "tx_ms": float(tx_ms),
        "network_delay_ms": float(network_delay_ms),
        "transfer_ms": float(transfer_ms),
    }
