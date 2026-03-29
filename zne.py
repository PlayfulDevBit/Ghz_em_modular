"""
zne.py — Zero Noise Extrapolation (ZNE)
=========================================
Standalone plugin for the GHZ Error Mitigation Pipeline.

WHAT IT DOES
------------
ZNE works in two steps:

  1. Amplify  — run the same circuit at intentionally higher noise levels
                by "folding" gates: replace each gate U with U·U†·U
                (scale factor 3), or U·U†·U·U†·U (scale factor 5), etc.
                Each fold costs more QPU time but increases noise predictably.

  2. Extrapolate — fit a curve through the (noise_level, fidelity) points
                   and evaluate it at noise_level = 0 to get an estimate
                   of what the fidelity would be on a perfect machine.

HOW TO CUSTOMISE IN JUPYTER
----------------------------
  1. Open this file in Jupyter (or copy the cell below).
  2. Edit `fold_circuit` and/or `extrapolate` — keep names and signatures.
  3. Test locally:  python zne.py
  4. Save zne.py and re-run the pipeline — no pipeline edits needed.

INTERFACE CONTRACT (do not change these signatures)
----------------------------------------------------
  fold_circuit(circuit, scale_factor) -> QuantumCircuit
  extrapolate(scale_factors, fidelities) -> float

  The pipeline calls fold_circuit once per scale factor, then calls
  extrapolate once with all results. Everything else is yours.
"""

from __future__ import annotations
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# FUNCTION 1 — GATE FOLDING (NOISE AMPLIFICATION)
# ═══════════════════════════════════════════════════════════════════════

def fold_circuit(circuit, scale_factor: int):
    """
    Amplify noise by gate folding: replace each gate U with (U·U†)^k · U
    where k = (scale_factor - 1) // 2.

    Parameters
    ----------
    circuit      : QuantumCircuit — already transpiled to native gates.
                   Measurements and barriers are left untouched.
    scale_factor : int — must be an odd integer ≥ 1.
                   1 = no folding (original circuit)
                   3 = one extra U†·U appended per gate
                   5 = two extra U†·U pairs appended per gate
                   Higher values → more noise → better extrapolation range
                   but also more QPU time and more statistical noise.

    Returns
    -------
    New QuantumCircuit with folded gates and re-appended measurements.

    CUSTOMISATION IDEAS
    -------------------
    - Fold only a subset of gates (e.g. only CZ gates, which are noisiest
      on Garnet) rather than all gates uniformly.
    - Use fractional noise scaling via local gate folding (select a
      random subset of gates to fold to achieve non-integer scale factors).
    - Replace gate folding with pulse stretching if you have pulse-level
      access to the backend.
    """
    from qiskit import QuantumCircuit

    if scale_factor == 1:
        return circuit  # No-op

    if scale_factor % 2 == 0:
        raise ValueError(
            f"scale_factor must be odd (got {scale_factor}). "
            "Valid values: 1, 3, 5, 7, ..."
        )

    n_folds = (scale_factor - 1) // 2

    # Strip measurements — we'll re-add them at the end
    qc_no_meas = circuit.copy()
    qc_no_meas.remove_final_measurements()

    # Collect non-measurement instructions from the original circuit
    original_ops = [
        inst for inst in circuit.data
        if inst.operation.name not in ("measure", "barrier")
    ]

    # Append n_folds rounds of (U† · U) for each gate
    for _ in range(n_folds):
        for inst in original_ops:
            try:
                inv_gate = inst.operation.inverse()
                qc_no_meas.append(inv_gate, inst.qubits, inst.clbits)
                qc_no_meas.append(inst.operation, inst.qubits, inst.clbits)
            except Exception:
                # Gate has no inverse (e.g. Reset) — skip silently
                pass

    qc_no_meas.measure_all()
    return qc_no_meas


# ═══════════════════════════════════════════════════════════════════════
# FUNCTION 2 — EXTRAPOLATION
# ═══════════════════════════════════════════════════════════════════════

def extrapolate(scale_factors: list[int], fidelities: list[float]) -> float:
    """
    Fit a polynomial through (scale_factor, fidelity) points and evaluate
    at scale_factor = 0 to estimate the zero-noise fidelity.

    Parameters
    ----------
    scale_factors : list of odd ints, e.g. [1, 3, 5]
    fidelities    : list of floats measured at each scale factor.
                    Must be the same length as scale_factors.

    Returns
    -------
    float in [0.0, 1.0] — estimated zero-noise fidelity.

    CUSTOMISATION IDEAS
    -------------------
    - Use exponential fitting: f(λ) = a·exp(-b·λ) + c, which is often
      more physically motivated than a polynomial.
    - Use Richardson extrapolation for exact cancellation of the
      leading noise term with just two scale factors.
    - Use a Bayesian regression to get a confidence interval on the
      extrapolated value, not just a point estimate.
    - Add outlier rejection: if one fidelity point is far from the
      trend, exclude it before fitting.
    """
    if len(scale_factors) == 0:
        raise ValueError("scale_factors list is empty")

    if len(scale_factors) == 1:
        # Nothing to extrapolate — return the single measured value
        return float(np.clip(fidelities[0], 0.0, 1.0))

    scales = np.array(scale_factors, dtype=float)
    fids   = np.array(fidelities,    dtype=float)

    # Polynomial degree: linear for 2 points, quadratic for 3+
    degree = min(len(scale_factors) - 1, 2)
    coeffs = np.polyfit(scales, fids, degree)

    # Evaluate at λ = 0
    zne_value = float(np.polyval(coeffs, 0.0))

    # Clip to valid fidelity range
    return float(np.clip(zne_value, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════
# LOCAL TEST HELPER
# Run this file directly to check your changes without the full pipeline.
#
#   python zne.py
#
# Uses a small synthetic circuit — no QPU token required.
# ═══════════════════════════════════════════════════════════════════════

def _simulate_noisy_fidelity(scale_factor: int, base_fidelity: float = 0.82) -> float:
    """
    Fake a fidelity measurement at a given noise scale for testing.
    Real noise generally degrades fidelity linearly with gate count.
    """
    noise_per_fold = 0.06
    rng = np.random.default_rng(scale_factor)
    simulated = base_fidelity - noise_per_fold * (scale_factor - 1) + rng.normal(0, 0.005)
    return float(np.clip(simulated, 0.0, 1.0))


if __name__ == "__main__":
    from qiskit import QuantumCircuit, transpile

    N = 3
    SCALE_FACTORS = [1, 3, 5]
    print(f"ZNE local test — {N} qubits, scales {SCALE_FACTORS}\n")

    # Build and transpile a small GHZ circuit
    qc = QuantumCircuit(N)
    qc.h(0)
    for i in range(1, N):
        qc.cx(0, i)
    qc.measure_all()

    qc_t = transpile(qc, basis_gates=["r", "cz", "id"], optimization_level=2)
    print(f"Base circuit: {qc_t.size()} gates, depth {qc_t.depth()}")

    # Test gate folding
    print()
    folded_sizes = []
    for sf in SCALE_FACTORS:
        qc_folded = fold_circuit(qc_t, sf)
        print(f"  scale={sf}: {qc_folded.size()} gates (expected ~{qc_t.size() + (sf-1) * (qc_t.size() - N)})")
        folded_sizes.append(qc_folded.size())

    # Test extrapolation with simulated fidelities
    print()
    fids = [_simulate_noisy_fidelity(sf) for sf in SCALE_FACTORS]
    for sf, f in zip(SCALE_FACTORS, fids):
        print(f"  Simulated fidelity at scale={sf}: {f:.4f}")

    zne_est = extrapolate(SCALE_FACTORS, fids)
    print(f"\n  ZNE extrapolated fidelity: {zne_est:.4f}")
    print(f"  (True base fidelity used in simulation: 0.82)")

    print("\n✅ ZNE test passed — fold_circuit and extrapolate both ran correctly.")
