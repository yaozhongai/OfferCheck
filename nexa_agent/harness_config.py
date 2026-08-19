"""Runtime feature switches used for fair Harness comparisons."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HarnessRuntimeConfig:
    evidence_gate: bool = True
    verifier: bool = True
    injection_guard: bool = True
    dynamic_upgrade: bool = True
    termination_guidance: bool = True
    risk_calibration: bool = True


FULL_HARNESS = HarnessRuntimeConfig()

MINIMAL_REACT = HarnessRuntimeConfig(
    evidence_gate=False,
    verifier=False,
    injection_guard=False,
    dynamic_upgrade=False,
    termination_guidance=False,
    risk_calibration=False,
)
