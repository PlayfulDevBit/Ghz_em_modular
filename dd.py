"""
dd.py — Dynamical Decoupling (DD)
===================================
Standalone plugin for the GHZ Error Mitigation Pipeline.

WHAT IT DOES
------------
Qubits accumulate errors while they sit idle waiting for other qubits to
finish their gates. Dynamical Decoupling inserts carefully timed X pulses
into those idle gaps. Because X·X = I, these pulses cancel each other out
logically but physically "refocus" the qubit and suppress decoherence.

The most common sequence is XX (two X gates), also called the Hahn echo.
More advanced sequences (XY4, CPMG) can suppress higher-order noise.

HOW TO CUSTOMISE IN JUPYTER
----------------------------
  1. Open this file in Jupyter (or copy the cell below).
  2. Edit the body of `apply_dd` — keep the function name and signature.
  3. Test locally:  python dd.py
  4. Save dd.py and re-run the pipeline — no pipeline edits needed.

INTERFACE CONTRACT (do not change this signature)
--------------------------------------------------
  apply_dd(circuit, backend) -> QuantumCircuit

  Input  : transpiled QuantumCircuit + backend
  Output : new QuantumCircuit with DD sequences inserted
  The pipeline only calls apply_dd; everything else is yours.
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════
# FUNCTION — APPLY DYNAMICAL DECOUPLING
# ═══════════════════════════════════════════════════════════════════════

def apply_dd(circuit, backend) -> object:
    """
    Insert dynamical decoupling sequences into idle qubit slots.

    Parameters
    ----------
    circuit : QuantumCircuit
        A transpiled circuit (already mapped to backend native gates).
        Do NOT pass an un-transpiled circuit — DD needs to know the
        actual gate timing to find the idle windows.
    backend : Qiskit backend
        The target backend, used to read gate timing (backend.target).

    Returns
    -------
    QuantumCircuit with DD pulses inserted.
    If DD fails for any reason (unsupported backend target, scheduling
    error) the original circuit is returned unchanged — the pipeline
    will continue with a warning rather than crashing.

    CUSTOMISATION IDEAS
    -------------------
    - Change `dd_sequence` to try different pulse sequences:
        [XGate(), XGate()]             — XX / Hahn echo (default)
        [XGate(), YGate(), XGate(), YGate()]  — XY4 sequence
        [XGate(), XGate(), XGate(), XGate()]  — CPMG-style
    - Pass `spacing` to PadDynamicalDecoupling to control pulse placement.
    - Use `pulse_alignment` for backends with strict timing constraints.
    - For IQM Garnet specifically, check available native gates before
      inserting sequences (Garnet uses R and CZ gates natively).
    """
    from qiskit.circuit.library import XGate
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import (
        ALAPScheduleAnalysis,
        PadDynamicalDecoupling,
    )

    # ── Choose your DD sequence here ──────────────────────────────────
    dd_sequence = [XGate(), XGate()]   # XX / Hahn echo

    try:
        pm = PassManager([
            ALAPScheduleAnalysis(target=backend.target),
            PadDynamicalDecoupling(target=backend.target, dd_sequence=dd_sequence),
        ])
        circuit_dd = pm.run(circuit)
        return circuit_dd

    except Exception as exc:
        # Graceful fallback: return circuit unchanged so the pipeline
        # can continue. The pipeline task will log this warning.
        import warnings
        warnings.warn(
            f"[DD] PassManager failed ({exc}). "
            "Returning circuit without DD. "
            "Check that the backend supports gate-level scheduling.",
            stacklevel=2,
        )
        return circuit


# ═══════════════════════════════════════════════════════════════════════
# LOCAL TEST HELPER
# Run this file directly to check your changes without the full pipeline.
#
#   python dd.py
#
# Requires a real backend token OR uses a minimal stub that just checks
# the function runs without crashing (no QPU calls needed for that).
# ═══════════════════════════════════════════════════════════════════════

def _build_test_circuit(n: int):
    """Build a small GHZ circuit for testing."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    qc.measure_all()
    return qc


class _FakeTarget:
    """Minimal stub so PassManager doesn't crash without a real backend."""
    operation_names = ["r", "cz", "id", "measure", "barrier"]

    def operation_names_for_qargs(self, *args):
        return self.operation_names


class _FakeBackend:
    target = _FakeTarget()


if __name__ == "__main__":
    from qiskit import transpile

    N = 3
    print(f"DD local test — {N} qubits, generic basis gates\n")

    qc = _build_test_circuit(N)
    print(f"Original circuit: {qc.size()} gates, depth {qc.depth()}")

    # Transpile to generic basis (no real backend needed)
    qc_t = transpile(qc, basis_gates=["r", "cz", "id"], optimization_level=2)
    print(f"Transpiled: {qc_t.size()} gates, depth {qc_t.depth()}")

    # Try DD with fake backend (PassManager will likely fall through to
    # the graceful fallback — that's expected without real timing data)
    print("\nApplying DD (may fall back gracefully without real backend)...")
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        qc_dd = apply_dd(qc_t, _FakeBackend())
        if w:
            print(f"  Warning: {w[-1].message}")
        else:
            print(f"  DD circuit: {qc_dd.size()} gates, depth {qc_dd.depth()}")

    print("\n✅ DD test passed — function ran without crashing.")
    print("   For real DD validation, run with a live IQM Garnet backend.")
