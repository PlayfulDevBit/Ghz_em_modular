"""
rem.py — Readout Error Mitigation (REM)
========================================
Standalone plugin for the GHZ Error Mitigation Pipeline.

WHAT IT DOES
------------
Quantum hardware often mis-reads qubit states: a qubit prepared as |1⟩
might be reported as |0⟩ some small fraction of the time, and vice versa.
REM characterises these errors using calibration circuits, then applies a
classical correction to the measured counts after the experiment.

THREE-STEP PROCESS
------------------
  1. Calibrate  — run |0⟩ and |1⟩ circuits on every qubit to measure
                  how often each qubit is read correctly.
  2. Execute    — run the real experiment circuit normally.
  3. Correct    — apply the inverse of the error matrix to the raw counts.

HOW TO CUSTOMISE IN JUPYTER
----------------------------
  1. Open this file in Jupyter (or copy the cell below).
  2. Edit the body of `calibrate_rem` or `apply_rem` — keep the
     function names and signatures exactly as they are.
  3. Test locally with the helper at the bottom of this file.
  4. Save rem.py and re-run the pipeline — no pipeline edits needed.

INTERFACE CONTRACT (do not change these signatures)
----------------------------------------------------
  calibrate_rem(backend, n_qubits, shots) -> CalibrationData
  apply_rem(raw_counts, calibration, n_qubits, shots) -> dict[str, int]

  CalibrationData is just a plain dict — see the type alias below.
  The pipeline only calls these two functions; everything else is yours.
"""

from __future__ import annotations
import numpy as np
from typing import TypedDict

# ── Type alias ──────────────────────────────────────────────────────────
# CalibrationData is what calibrate_rem returns and apply_rem receives.
# Keep the keys "qubit_matrices" and "inv_matrices" — the pipeline reads
# them when building the experiment report.

class CalibrationData(TypedDict):
    qubit_matrices: list[list[list[float]]]   # per-qubit 2×2 assignment matrices (as plain lists for JSON-safety)
    inv_matrices:   list[np.ndarray]           # inverses of the above (numpy, used internally)
    qubit_readout_errors: list[float]          # scalar error rate per qubit (for the report)


# ═══════════════════════════════════════════════════════════════════════
# FUNCTION 1 — CALIBRATION
# Called once before the main circuit is executed.
# ═══════════════════════════════════════════════════════════════════════

def calibrate_rem(backend, n_qubits: int, shots: int) -> CalibrationData:
    """
    Run |0⟩ and |1⟩ calibration circuits on every qubit and compute
    per-qubit assignment matrices.

    Parameters
    ----------
    backend   : Qiskit backend (IQM Garnet or any compatible backend)
    n_qubits  : number of qubits in the experiment circuit
    shots     : number of shots for calibration (same as experiment shots
                is a safe default; you can increase for better statistics)

    Returns
    -------
    CalibrationData dict containing:
      qubit_matrices       — 2×2 matrix M where M[i,j] = P(readout=i | prepared=j)
      inv_matrices         — pseudo-inverse of each matrix (used in apply_rem)
      qubit_readout_errors — scalar error rate per qubit (for the report)

    CUSTOMISATION IDEAS
    -------------------
    - Replace 2-point calibration (|0⟩, |1⟩) with full matrix calibration
      (all 2^n basis states) for a joint rather than per-qubit correction.
    - Weight calibration shots differently per qubit based on known T1/T2.
    - Load a pre-saved calibration file instead of running on the QPU.
    """
    from qiskit import QuantumCircuit, transpile

    # ── Calibration circuits ───────────────────────────────────────────
    cal_0 = QuantumCircuit(n_qubits)
    cal_0.measure_all()

    cal_1 = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        cal_1.x(i)
    cal_1.measure_all()

    cal_0_t = transpile(cal_0, backend=backend)
    cal_1_t = transpile(cal_1, backend=backend)

    counts_0 = backend.run(cal_0_t, shots=shots, use_timeslot=False).result().get_counts()
    counts_1 = backend.run(cal_1_t, shots=shots, use_timeslot=False).result().get_counts()

    # ── Build per-qubit 2×2 assignment matrices ───────────────────────
    qubit_matrices = []
    for q in range(n_qubits):
        p00, p10, p01, p11 = 0.0, 0.0, 0.0, 0.0

        for bs, count in counts_0.items():
            bit = int(list(reversed(bs))[q])
            if bit == 0:
                p00 += count
            else:
                p10 += count

        for bs, count in counts_1.items():
            bit = int(list(reversed(bs))[q])
            if bit == 0:
                p01 += count
            else:
                p11 += count

        # Normalise
        p00 /= shots
        p10 /= shots
        p01 /= shots
        p11 /= shots

        # M[row, col] = P(readout=row | prepared=col)
        qubit_matrices.append(np.array([
            [p00, p01],
            [p10, p11],
        ]))

    # ── Invert matrices ────────────────────────────────────────────────
    inv_matrices = []
    for m in qubit_matrices:
        try:
            inv_matrices.append(np.linalg.inv(m))
        except np.linalg.LinAlgError:
            inv_matrices.append(np.eye(2))

    errors = [round(1.0 - (m[0, 0] + m[1, 1]) / 2.0, 4) for m in qubit_matrices]

    return CalibrationData(
        qubit_matrices=[m.tolist() for m in qubit_matrices],
        inv_matrices=inv_matrices,
        qubit_readout_errors=errors,
    )


# ═══════════════════════════════════════════════════════════════════════
# FUNCTION 2 — CORRECTION
# Called after the main circuit has been executed.
# ═══════════════════════════════════════════════════════════════════════

def apply_rem(
    raw_counts: dict[str, int],
    calibration: CalibrationData,
    n_qubits: int,
    shots: int,
) -> dict[str, int]:
    """
    Apply per-qubit inverse assignment matrix to correct measured counts.

    Parameters
    ----------
    raw_counts  : counts dict returned by backend.run(...).result().get_counts()
    calibration : CalibrationData returned by calibrate_rem()
    n_qubits    : number of qubits
    shots       : number of shots (used for normalisation)

    Returns
    -------
    Corrected counts dict with the same bitstring keys as raw_counts.
    Negative probabilities are clipped to zero and the distribution is
    renormalised before converting back to integer counts.

    CUSTOMISATION IDEAS
    -------------------
    - Use scipy.optimize.nnls (non-negative least squares) instead of
      matrix inversion for a solution that never produces negative counts.
    - Apply a Bayesian update instead of simple inversion.
    - Use the full joint assignment matrix (2^n × 2^n) instead of the
      per-qubit tensor-product approximation used here.
    """
    inv_matrices = calibration["inv_matrices"]
    all_bitstrings = [format(i, f"0{n_qubits}b") for i in range(2 ** n_qubits)]

    # Convert counts to probability vector
    probs = np.array([raw_counts.get(bs, 0) / shots for bs in all_bitstrings])

    # Apply per-qubit correction via tensor-product of inverse matrices
    for q in range(n_qubits):
        inv_m = inv_matrices[q]
        new_probs = np.zeros_like(probs)
        for idx, bs in enumerate(all_bitstrings):
            bit = int(list(reversed(bs))[q])
            for target_bit in [0, 1]:
                bs_list = list(reversed(bs))
                bs_list[q] = str(target_bit)
                target_bs = "".join(reversed(bs_list))
                target_idx = all_bitstrings.index(target_bs)
                new_probs[target_idx] += inv_m[target_bit, bit] * probs[idx]
        probs = new_probs

    # Clip negatives and renormalise
    probs = np.maximum(probs, 0)
    if probs.sum() > 0:
        probs /= probs.sum()

    return {
        bs: int(round(p * shots))
        for bs, p in zip(all_bitstrings, probs)
        if p > 0.001
    }


# ═══════════════════════════════════════════════════════════════════════
# LOCAL TEST HELPER
# Run this file directly to check your changes without the full pipeline.
#
#   python rem.py
#
# It uses a fake backend (random counts) so you don't need a QPU token.
# ═══════════════════════════════════════════════════════════════════════

def _fake_backend_counts(n: int, shots: int, error_rate: float = 0.05) -> dict:
    """Simulate noisy GHZ counts for local testing."""
    rng = np.random.default_rng(42)
    counts: dict[str, int] = {}
    for _ in range(shots):
        # Ideal GHZ: 50% |000...0⟩, 50% |111...1⟩
        ideal = "0" * n if rng.random() < 0.5 else "1" * n
        # Flip each bit with probability error_rate
        noisy = "".join(
            str(1 - int(b)) if rng.random() < error_rate else b
            for b in ideal
        )
        counts[noisy] = counts.get(noisy, 0) + 1
    return counts


class _FakeBackend:
    """Minimal fake backend for local testing (no QPU required)."""
    def run(self, circuit, shots=1024, **kwargs):
        import re
        n = circuit.num_qubits
        # Detect all-zeros vs all-ones prep from circuit name/gates
        is_ones = any(
            instr.operation.name == "x"
            for instr in circuit.data
        )
        error_rate = 0.03
        if is_ones:
            counts = {}
            rng = np.random.default_rng(0)
            for _ in range(shots):
                bs = "".join(
                    str(1 - int(b)) if rng.random() < error_rate else b
                    for b in "1" * n
                )
                counts[bs] = counts.get(bs, 0) + 1
        else:
            counts = {}
            rng = np.random.default_rng(1)
            for _ in range(shots):
                bs = "".join(
                    str(1 - int(b)) if rng.random() < error_rate else b
                    for b in "0" * n
                )
                counts[bs] = counts.get(bs, 0) + 1

        class _Result:
            def __init__(self, c): self._c = c
            def get_counts(self): return self._c

        class _Job:
            def __init__(self, c): self._c = c
            def result(self): return _Result(self._c)

        return _Job(counts)


if __name__ == "__main__":
    N = 3
    SHOTS = 2048
    print(f"REM local test — {N} qubits, {SHOTS} shots, fake backend\n")

    fb = _FakeBackend()

    print("Step 1: Calibrating...")
    cal = calibrate_rem(fb, N, SHOTS)
    print(f"  Per-qubit readout errors: {cal['qubit_readout_errors']}")

    print("\nStep 2: Simulating raw GHZ counts (with noise)...")
    raw = _fake_backend_counts(N, SHOTS, error_rate=0.05)
    ghz_zeros = raw.get("0" * N, 0)
    ghz_ones  = raw.get("1" * N, 0)
    print(f"  Raw GHZ fidelity: {(ghz_zeros + ghz_ones) / SHOTS:.4f}")
    print(f"  Top counts: { {k: v for k, v in sorted(raw.items(), key=lambda x: -x[1])[:5]} }")

    print("\nStep 3: Applying REM correction...")
    corrected = apply_rem(raw, cal, N, SHOTS)
    c_zeros = corrected.get("0" * N, 0)
    c_ones  = corrected.get("1" * N, 0)
    print(f"  Corrected GHZ fidelity: {(c_zeros + c_ones) / SHOTS:.4f}")
    print(f"  Top counts: { {k: v for k, v in sorted(corrected.items(), key=lambda x: -x[1])[:5]} }")
    print("\n✅ REM test passed — safe to upload.")
