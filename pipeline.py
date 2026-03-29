"""
GHZ Error Mitigation Pipeline · IQM Garnet
============================================
Single configurable pipeline with toggleable error reduction techniques.
Runs on real IQM Garnet QPU. Produces rich SVG comparison artifacts.

Techniques are implemented in separate plugin files:
  dd.py  — Dynamical Decoupling
  rem.py — Readout Error Mitigation
  zne.py — Zero Noise Extrapolation

To customise a technique, edit its plugin file and re-run the pipeline.
See README.md for the full workflow.

Local:
    python ghz_mitigation_pipeline.py
    python ghz_mitigation_pipeline.py --no-dd --no-zne   # REM only

Serverless:
    python deploy_mitigation.py          # register deployment
    # then trigger from Prefect Cloud UI with parameter toggles
"""

import time
import argparse
import numpy as np
from datetime import datetime, timezone

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

# ── Plugin imports ──────────────────────────────────────────────────────
# Each technique lives in its own file. The pipeline only calls the
# public functions; all implementation details are inside the plugins.
from dd  import apply_dd
from rem import calibrate_rem, apply_rem
from zne import fold_circuit, extrapolate


# ═══════════════════════════════════════════════════════════════════════
# TOKEN + BACKEND HELPERS
# ═══════════════════════════════════════════════════════════════════════

def get_iqm_token() -> str:
    try:
        from prefect.blocks.system import Secret
        return Secret.load("iqm-resonance-token").get()
    except Exception:
        return ""


def get_iqm_backend(token: str):
    from iqm.qiskit_iqm import IQMProvider
    return IQMProvider(
        "https://cocos.resonance.meetiqm.com/garnet",
        token=token,
    ).get_backend()


def _ghz_fidelity(counts: dict, n: int, shots: int) -> float:
    zeros = counts.get("0" * n, 0)
    ones  = counts.get("1" * n, 0)
    total = sum(counts.values()) or shots
    return round((zeros + ones) / total, 4)


# ═══════════════════════════════════════════════════════════════════════
# STAGE 1 — CIRCUIT CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

@task(name="1 · Build & Transpile GHZ Circuit", tags=["stage:1", "infra:cpu"])
def build_and_transpile(num_qubits: int) -> dict:
    logger = get_run_logger()
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for i in range(1, num_qubits):
        qc.cx(0, i)
    qc.measure_all()

    logger.info(f"GHZ-{num_qubits}: {qc.size()} gates, depth {qc.depth()}")

    token = get_iqm_token()
    if token:
        backend = get_iqm_backend(token)
        qc_t = transpile(qc, backend=backend, optimization_level=2)
        logger.info(f"Transpiled for IQM Garnet: {qc_t.size()} gates, depth {qc_t.depth()}")
    else:
        qc_t = transpile(qc, basis_gates=["r", "cz", "id"], optimization_level=2)
        logger.info(f"Transpiled (generic): {qc_t.size()} gates, depth {qc_t.depth()}")

    return {
        "num_qubits": num_qubits,
        "original_gates": qc.size(),
        "original_depth": qc.depth(),
        "transpiled_gates": qc_t.size(),
        "transpiled_depth": qc_t.depth(),
        "_circuit": qc,
        "_transpiled": qc_t,
    }


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2 — BASELINE EXECUTION
# ═══════════════════════════════════════════════════════════════════════

@task(name="2 · BASELINE Execution", tags=["stage:2", "infra:qpu"], retries=2, retry_delay_seconds=10)
def run_baseline(circuit_data: dict, shots: int) -> dict:
    logger = get_run_logger()
    token = get_iqm_token()
    if not token:
        raise RuntimeError("IQM token required — set Prefect Secret 'iqm-resonance-token'")

    from qiskit import transpile
    backend = get_iqm_backend(token)
    qc_t = transpile(circuit_data["_circuit"], backend=backend, optimization_level=2)

    logger.info(f"BASELINE: submitting {shots} shots...")
    t0 = time.time()
    counts = backend.run(qc_t, shots=shots, use_timeslot=False).result().get_counts()
    exec_time = round(time.time() - t0, 2)

    fidelity = _ghz_fidelity(counts, circuit_data["num_qubits"], shots)
    logger.info(f"BASELINE fidelity: {fidelity:.4f} ({exec_time}s)")

    return {
        "technique": "Baseline (raw)",
        "counts": counts,
        "fidelity": fidelity,
        "exec_time_s": exec_time,
        "shots": shots,
        "color": "#8B8B8B",
        "description": "No error mitigation applied",
    }


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3a — DYNAMICAL DECOUPLING
# Delegates entirely to dd.apply_dd()
# ═══════════════════════════════════════════════════════════════════════

@task(name="3a · Dynamical Decoupling (DD)", tags=["stage:3", "infra:qpu", "technique:dd"],
      retries=2, retry_delay_seconds=10)
def run_with_dd(circuit_data: dict, shots: int) -> dict:
    """
    Apply DD via dd.apply_dd(), then run on the QPU.
    To change the DD strategy, edit dd.py — not this task.
    """
    logger = get_run_logger()
    token = get_iqm_token()
    if not token:
        raise RuntimeError("IQM token required")

    from qiskit import transpile
    backend = get_iqm_backend(token)
    qc_t = transpile(circuit_data["_circuit"], backend=backend, optimization_level=2)

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        qc_dd = apply_dd(qc_t, backend)
        if w:
            logger.warning(f"DD fallback: {w[-1].message}")
        else:
            logger.info(f"DD applied: {qc_dd.size()} gates (was {qc_t.size()})")

    t0 = time.time()
    counts = backend.run(qc_dd, shots=shots, use_timeslot=False).result().get_counts()
    exec_time = round(time.time() - t0, 2)

    fidelity = _ghz_fidelity(counts, circuit_data["num_qubits"], shots)
    logger.info(f"DD fidelity: {fidelity:.4f} ({exec_time}s)")

    return {
        "technique": "Dynamical Decoupling (DD)",
        "counts": counts,
        "fidelity": fidelity,
        "exec_time_s": exec_time,
        "shots": shots,
        "color": "#2196F3",
        "description": "XX pulse sequences on idle qubits suppress decoherence",
    }


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3b — READOUT ERROR MITIGATION
# Delegates to rem.calibrate_rem() + rem.apply_rem()
# ═══════════════════════════════════════════════════════════════════════

@task(name="3b · Readout Error Mitigation (REM)", tags=["stage:3", "infra:qpu", "technique:rem"],
      retries=2, retry_delay_seconds=10)
def run_with_rem(circuit_data: dict, shots: int) -> dict:
    """
    Calibrate and correct via rem.calibrate_rem() + rem.apply_rem().
    To change the calibration or correction strategy, edit rem.py.
    """
    logger = get_run_logger()
    token = get_iqm_token()
    if not token:
        raise RuntimeError("IQM token required")

    from qiskit import transpile
    backend = get_iqm_backend(token)
    n = circuit_data["num_qubits"]

    logger.info("REM: Running calibration circuits...")
    calibration = calibrate_rem(backend, n, shots)
    for q, err in enumerate(calibration["qubit_readout_errors"]):
        logger.info(f"  Q{q}: readout_error = {err:.4f}")

    qc_t = transpile(circuit_data["_circuit"], backend=backend, optimization_level=2)
    t0 = time.time()
    raw_counts = backend.run(qc_t, shots=shots, use_timeslot=False).result().get_counts()
    exec_time = round(time.time() - t0, 2)

    corrected_counts = apply_rem(raw_counts, calibration, n, shots)
    fidelity = _ghz_fidelity(corrected_counts, n, shots)
    logger.info(f"REM fidelity: {fidelity:.4f} ({exec_time}s + calibration)")

    return {
        "technique": "Readout Error Mitigation (REM)",
        "counts": corrected_counts,
        "raw_counts": raw_counts,
        "fidelity": fidelity,
        "exec_time_s": exec_time,
        "shots": shots,
        "qubit_readout_errors": calibration["qubit_readout_errors"],
        "qubit_matrices": calibration["qubit_matrices"],
        "color": "#4CAF50",
        "description": "Calibration-based measurement error correction",
    }


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3c — ZERO NOISE EXTRAPOLATION
# Delegates to zne.fold_circuit() + zne.extrapolate()
# ═══════════════════════════════════════════════════════════════════════

@task(name="3c · Zero Noise Extrapolation (ZNE)", tags=["stage:3", "infra:qpu", "technique:zne"],
      retries=2, retry_delay_seconds=10)
def run_with_zne(circuit_data: dict, shots: int, scale_factors: list[int]) -> dict:
    """
    Run at multiple noise levels via zne.fold_circuit(), then extrapolate
    via zne.extrapolate(). To change folding or fitting strategy, edit zne.py.
    """
    logger = get_run_logger()
    token = get_iqm_token()
    if not token:
        raise RuntimeError("IQM token required")

    from qiskit import transpile
    backend = get_iqm_backend(token)
    n = circuit_data["num_qubits"]
    qc_t = transpile(circuit_data["_circuit"], backend=backend, optimization_level=2)

    fidelities_at_scales = []
    counts_at_scale_1 = {}

    for scale in scale_factors:
        qc_folded = fold_circuit(qc_t, scale)
        logger.info(f"ZNE scale={scale}: {qc_folded.size()} gates")

        t0 = time.time()
        counts = backend.run(qc_folded, shots=shots, use_timeslot=False).result().get_counts()
        exec_time = round(time.time() - t0, 2)

        f = _ghz_fidelity(counts, n, shots)
        fidelities_at_scales.append(f)
        logger.info(f"  Fidelity at scale {scale}: {f:.4f} ({exec_time}s)")

        if scale == 1:
            counts_at_scale_1 = counts

    zne_fidelity = extrapolate(scale_factors, fidelities_at_scales)
    extrapolation_type = "polynomial" if len(scale_factors) > 2 else "linear"
    logger.info(f"ZNE extrapolated fidelity: {zne_fidelity:.4f} ({extrapolation_type})")

    return {
        "technique": "Zero Noise Extrapolation (ZNE)",
        "counts": counts_at_scale_1 or counts,
        "fidelity": round(zne_fidelity, 4),
        "exec_time_s": len(scale_factors),
        "shots": shots * len(scale_factors),
        "scale_factors": scale_factors,
        "fidelities_at_scales": [round(f, 4) for f in fidelities_at_scales],
        "extrapolation_type": extrapolation_type,
        "color": "#FF9800",
        "description": f"Gate folding at scales {scale_factors}, {extrapolation_type} extrapolation to zero noise",
    }


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3d — COMBINED (all enabled techniques)
# Imports from all three plugins and chains them.
# ═══════════════════════════════════════════════════════════════════════

@task(name="3d · Combined (DD + REM + ZNE)", tags=["stage:3", "infra:qpu", "technique:combined"],
      retries=2, retry_delay_seconds=10)
def run_combined(circuit_data: dict, shots: int,
                 enable_dd: bool, enable_rem: bool, enable_zne: bool,
                 zne_scale_factors: list[int]) -> dict:
    """
    Run all enabled techniques together in one task, chaining them in order:
      1. Transpile
      2. Apply DD (if enabled)         — via dd.apply_dd()
      3. Calibrate REM (if enabled)    — via rem.calibrate_rem()
      4. For each ZNE scale (or just 1):
           a. Fold circuit              — via zne.fold_circuit()
           b. Run on QPU
           c. Apply REM correction      — via rem.apply_rem()
           d. Record fidelity
      5. Extrapolate                   — via zne.extrapolate()
    """
    logger = get_run_logger()
    token = get_iqm_token()
    if not token:
        raise RuntimeError("IQM token required")

    from qiskit import QuantumCircuit, transpile
    import warnings

    backend = get_iqm_backend(token)
    n = circuit_data["num_qubits"]

    qc_t = transpile(circuit_data["_circuit"], backend=backend, optimization_level=2)

    # ── Step 1: DD ──────────────────────────────────────────────────────
    if enable_dd:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            qc_t = apply_dd(qc_t, backend)
            if w:
                logger.warning(f"Combined DD fallback: {w[-1].message}")
            else:
                logger.info("Combined: DD applied")

    # ── Step 2: REM calibration ─────────────────────────────────────────
    calibration = None
    if enable_rem:
        logger.info("Combined: Running REM calibration...")
        calibration = calibrate_rem(backend, n, shots)
        logger.info("Combined: REM calibration done")

    # ── Step 3: ZNE loop (or single run) ───────────────────────────────
    scales = zne_scale_factors if enable_zne else [1]
    fidelities = []
    last_counts = {}

    for scale in scales:
        qc_run = fold_circuit(qc_t, scale)
        counts = backend.run(qc_run, shots=shots, use_timeslot=False).result().get_counts()

        if calibration is not None:
            counts = apply_rem(counts, calibration, n, shots)

        f = _ghz_fidelity(counts, n, shots)
        fidelities.append(f)
        last_counts = counts
        logger.info(f"Combined scale={scale}: fidelity={f:.4f}")

    # ── Step 4: Extrapolate ─────────────────────────────────────────────
    if enable_zne and len(scales) >= 2:
        combined_fidelity = extrapolate(scales, fidelities)
    else:
        combined_fidelity = fidelities[0]

    active = [t for t, on in [("DD", enable_dd), ("REM", enable_rem), ("ZNE", enable_zne)] if on]
    logger.info(f"COMBINED ({'+'.join(active)}): fidelity={combined_fidelity:.4f}")

    return {
        "technique": f"Combined ({'+'.join(active)})",
        "counts": last_counts,
        "fidelity": round(combined_fidelity, 4),
        "exec_time_s": 0,
        "shots": shots * len(scales),
        "active_techniques": active,
        "color": "#E91E63",
        "description": f"All enabled techniques applied together: {', '.join(active)}",
    }


# ═══════════════════════════════════════════════════════════════════════
# STAGE 4 — ARTIFACTS
# (unchanged from original — purely reporting, no technique logic)
# ═══════════════════════════════════════════════════════════════════════

@task(name="4.1 · Fidelity Comparison Chart", tags=["stage:4", "reporting"])
def publish_fidelity_chart(all_results: list[dict]) -> None:
    width = 600
    bar_height = 50
    gap = 15
    left_margin = 220
    right_margin = 80
    chart_width = width - left_margin - right_margin
    height = len(all_results) * (bar_height + gap) + 80

    bars_svg = ""
    for i, r in enumerate(all_results):
        y = 40 + i * (bar_height + gap)
        bar_w = max(2, r["fidelity"] * chart_width)
        color = r.get("color", "#888")
        bars_svg += (
            f'<text x="{left_margin - 10}" y="{y + bar_height/2 + 5}" '
            f'text-anchor="end" font-family="monospace" font-size="13" fill="#333">'
            f'{r["technique"]}</text>\n'
            f'<rect x="{left_margin}" y="{y}" width="{bar_w}" height="{bar_height}" '
            f'fill="{color}" rx="4" opacity="0.85"/>\n'
            f'<text x="{left_margin + bar_w + 8}" y="{y + bar_height/2 + 5}" '
            f'font-family="monospace" font-size="14" font-weight="bold" fill="{color}">'
            f'{r["fidelity"]:.4f}</text>\n'
        )

    ideal_x = left_margin + chart_width
    bars_svg += (
        f'<line x1="{ideal_x}" y1="30" x2="{ideal_x}" y2="{height - 20}" '
        f'stroke="#00C853" stroke-width="2" stroke-dasharray="6,4"/>\n'
        f'<text x="{ideal_x}" y="25" text-anchor="middle" font-family="monospace" '
        f'font-size="11" fill="#00C853">ideal=1.0</text>\n'
    )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="{width}" height="{height}" fill="#FAFAFA" rx="8"/>\n'
        f'<text x="{width/2}" y="20" text-anchor="middle" font-family="Arial" font-size="16" '
        f'font-weight="bold" fill="#333">GHZ Fidelity by Error Mitigation Technique</text>\n'
        f'{bars_svg}</svg>'
    )
    create_markdown_artifact(
        key="fidelity-comparison-chart",
        markdown=f"# Fidelity Comparison\n\n{svg}",
        description="Bar chart comparing GHZ fidelity across error mitigation techniques",
    )


@task(name="4.2 · Measurement Distribution Heatmap", tags=["stage:4", "reporting"])
def publish_heatmap(all_results: list[dict], num_qubits: int) -> None:
    n = num_qubits
    all_bitstrings = [format(i, f'0{n}b') for i in range(2**n)]
    techniques = [r["technique"] for r in all_results]

    cell_w = max(35, 500 // len(all_bitstrings))
    cell_h = 45
    left_margin = 230
    top_margin = 60
    width = left_margin + len(all_bitstrings) * cell_w + 80
    height = top_margin + len(techniques) * cell_h + 60

    cells_svg = ""
    for j, bs in enumerate(all_bitstrings):
        x = left_margin + j * cell_w + cell_w / 2
        highlight = bs == "0" * n or bs == "1" * n
        cells_svg += (
            f'<text x="{x}" y="{top_margin - 10}" text-anchor="middle" '
            f'font-family="monospace" font-size="11" '
            f'font-weight="{"bold" if highlight else "normal"}" '
            f'fill="{"#00C853" if highlight else "#666"}">|{bs}⟩</text>\n'
        )

    for i, r in enumerate(all_results):
        y = top_margin + i * cell_h
        total = sum(r["counts"].values()) or 1
        cells_svg += (
            f'<text x="{left_margin - 10}" y="{y + cell_h/2 + 4}" '
            f'text-anchor="end" font-family="monospace" font-size="12" fill="#333">'
            f'{r["technique"]}</text>\n'
        )
        for j, bs in enumerate(all_bitstrings):
            x = left_margin + j * cell_w
            prob = r["counts"].get(bs, 0) / total
            intensity = min(1.0, prob * 2)
            red   = int(255 * (1 - intensity))
            green = int(255 * (1 - intensity * 0.7))
            blue  = int(255 * (1 - intensity * 0.2))
            fill = f"rgb({red},{green},{blue})"
            cells_svg += (
                f'<rect x="{x}" y="{y}" width="{cell_w-1}" height="{cell_h-1}" '
                f'fill="{fill}" stroke="#ddd" stroke-width="0.5" rx="3"/>\n'
            )
            if prob > 0.01:
                text_color = "white" if intensity > 0.5 else "#333"
                cells_svg += (
                    f'<text x="{x + cell_w/2}" y="{y + cell_h/2 + 4}" '
                    f'text-anchor="middle" font-family="monospace" font-size="10" '
                    f'fill="{text_color}">{prob:.2f}</text>\n'
                )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="{width}" height="{height}" fill="#FAFAFA" rx="8"/>\n'
        f'<text x="{width/2}" y="20" text-anchor="middle" font-family="Arial" font-size="15" '
        f'font-weight="bold" fill="#333">Measurement Distribution Heatmap</text>\n'
        f'<text x="{width/2}" y="38" text-anchor="middle" font-family="Arial" font-size="11" '
        f'fill="#888">Color intensity = measurement probability. '
        f'Ideal GHZ: only |{"0"*n}⟩ and |{"1"*n}⟩ are populated.</text>\n'
        f'{cells_svg}</svg>'
    )
    create_markdown_artifact(
        key="measurement-heatmap",
        markdown=f"# Measurement Distribution Heatmap\n\n{svg}",
        description="2D heatmap: techniques × bitstrings, color = probability",
    )


@task(name="4.3 · ZNE Extrapolation Curve", tags=["stage:4", "reporting"])
def publish_zne_curve(zne_result: dict) -> None:
    if not zne_result or "scale_factors" not in zne_result:
        return

    scales = zne_result["scale_factors"]
    fids   = zne_result["fidelities_at_scales"]
    zne_fid = zne_result["fidelity"]

    width, height = 500, 350
    pl, pr, pt, pb = 70, 40, 50, 50
    pw = width - pl - pr
    ph = height - pt - pb
    max_scale = max(scales) + 0.5
    min_fid = min(min(fids), zne_fid) - 0.05
    max_fid = max(max(fids), zne_fid, 1.0) + 0.02

    def sx(v): return pl + (v / max_scale) * pw
    def sy(v): return pt + ph - ((v - min_fid) / (max_fid - min_fid + 1e-9)) * ph

    coeffs = np.polyfit(np.array(scales, dtype=float), np.array(fids), min(len(scales)-1, 2))
    line_pts = " ".join(
        f"{sx(lam):.1f},{sy(float(np.polyval(coeffs, lam))):.1f}"
        for lam in np.linspace(0, max_scale, 50)
    )

    svg_content = (
        f'<line x1="{pl}" y1="{pt}" x2="{pl}" y2="{pt+ph}" stroke="#999" stroke-width="1"/>\n'
        f'<line x1="{pl}" y1="{pt+ph}" x2="{pl+pw}" y2="{pt+ph}" stroke="#999" stroke-width="1"/>\n'
        f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">Noise Scale Factor (λ)</text>\n'
        f'<polyline points="{line_pts}" fill="none" stroke="#FF9800" stroke-width="2" stroke-dasharray="6,3"/>\n'
    )
    for s, f in zip(scales, fids):
        svg_content += (
            f'<circle cx="{sx(s)}" cy="{sy(f)}" r="7" fill="#FF9800" stroke="white" stroke-width="2"/>\n'
            f'<text x="{sx(s)}" y="{sy(f)-12}" text-anchor="middle" font-family="monospace" font-size="10" fill="#FF9800">{f:.3f}</text>\n'
        )
    svg_content += (
        f'<circle cx="{sx(0)}" cy="{sy(zne_fid)}" r="9" fill="#E91E63" stroke="white" stroke-width="2"/>\n'
        f'<text x="{sx(0)+15}" y="{sy(zne_fid)+4}" font-family="monospace" font-size="12" font-weight="bold" fill="#E91E63">ZNE = {zne_fid:.4f}</text>\n'
    )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'<rect width="{width}" height="{height}" fill="#FAFAFA" rx="8"/>\n'
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="Arial" font-size="15" font-weight="bold" fill="#333">ZNE Extrapolation Curve</text>\n'
        f'<text x="{width/2}" y="42" text-anchor="middle" font-family="Arial" font-size="11" fill="#888">Measured fidelity at amplified noise → extrapolated to zero noise</text>\n'
        f'{svg_content}</svg>'
    )
    create_markdown_artifact(
        key="zne-extrapolation-curve",
        markdown=f"# ZNE Extrapolation\n\n{svg}",
        description="Zero Noise Extrapolation: fidelity vs noise scale factor",
    )


@task(name="4.4 · REM Readout Error Map", tags=["stage:4", "reporting"])
def publish_rem_heatmap(rem_result: dict, num_qubits: int) -> None:
    if not rem_result or "qubit_matrices" not in rem_result:
        return

    matrices = rem_result["qubit_matrices"]
    errors   = rem_result["qubit_readout_errors"]
    n = num_qubits
    cell_size = 60
    qubit_block_w = cell_size * 2 + 20
    width = 150 + n * qubit_block_w + 40
    height = 250

    svg_content = ""
    for q in range(n):
        m = matrices[q]
        bx = 150 + q * qubit_block_w
        by = 60
        err = errors[q]
        err_color = "#4CAF50" if err < 0.02 else "#FF9800" if err < 0.05 else "#F44336"
        svg_content += (
            f'<text x="{bx + cell_size}" y="{by - 15}" text-anchor="middle" '
            f'font-family="monospace" font-size="13" font-weight="bold" fill="{err_color}">Q{q}</text>\n'
            f'<text x="{bx + cell_size}" y="{by - 2}" text-anchor="middle" '
            f'font-family="monospace" font-size="10" fill="{err_color}">err={err:.3f}</text>\n'
        )
        for row in range(2):
            for col in range(2):
                x = bx + col * cell_size
                y = by + row * cell_size
                val = m[row][col]
                if row == col:
                    intensity = val
                    r, g, b = int(255*(1-intensity)), 255, int(255*(1-intensity))
                else:
                    intensity = min(val * 10, 1)
                    r, g, b = 255, int(255*(1-intensity)), int(255*(1-intensity))
                svg_content += (
                    f'<rect x="{x}" y="{y}" width="{cell_size-2}" height="{cell_size-2}" '
                    f'fill="rgb({r},{g},{b})" stroke="#ccc" rx="4"/>\n'
                    f'<text x="{x + cell_size/2}" y="{y + cell_size/2 + 5}" '
                    f'text-anchor="middle" font-family="monospace" font-size="12" fill="#333">{val:.3f}</text>\n'
                )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="{width}" height="{height}" fill="#FAFAFA" rx="8"/>\n'
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="Arial" font-size="15" '
        f'font-weight="bold" fill="#333">Readout Error Assignment Matrices</text>\n'
        f'<text x="{width/2}" y="42" text-anchor="middle" font-family="Arial" font-size="11" '
        f'fill="#888">Diagonal = correct readout probability. Off-diagonal = error rate.</text>\n'
        f'{svg_content}</svg>'
    )
    create_markdown_artifact(
        key="rem-readout-error-map",
        markdown=f"# Readout Error Map\n\n{svg}",
        description="Per-qubit readout assignment matrices with error rates",
    )


@task(name="4.5 · Experiment Report", tags=["stage:4", "reporting"])
def publish_report(all_results: list[dict], circuit_data: dict,
                   enable_dd: bool, enable_rem: bool, enable_zne: bool) -> None:
    n = circuit_data["num_qubits"]
    baseline = all_results[0]

    rows = "\n".join(
        f"| {r['technique']} | {r['fidelity']:.4f} | "
        f"{((r['fidelity'] - baseline['fidelity']) / max(baseline['fidelity'], 0.001) * 100):+.1f}% | "
        f"{r['shots']:,} | {r.get('exec_time_s', 'N/A')} |"
        for r in all_results
    )
    best = max(all_results, key=lambda r: r["fidelity"])
    config_rows = (
        f"| Dynamical Decoupling (DD) | {'✅ Enabled' if enable_dd else '❌ Disabled'} |\n"
        f"| Readout Error Mitigation (REM) | {'✅ Enabled' if enable_rem else '❌ Disabled'} |\n"
        f"| Zero Noise Extrapolation (ZNE) | {'✅ Enabled' if enable_zne else '❌ Disabled'} |"
    )

    report = f"""# GHZ Error Mitigation Experiment Report

**Date**: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
**Backend**: IQM Garnet · **Circuit**: GHZ-{n}

## Configuration
| Technique | Status |
|-----------|--------|
{config_rows}

## Circuit Details
| Parameter | Value |
|-----------|-------|
| Qubits | {n} |
| Target state | (|{'0'*n}⟩ + |{'1'*n}⟩) / √2 |
| Original gates | {circuit_data['original_gates']} |
| Transpiled gates | {circuit_data['transpiled_gates']} |
| Transpiled depth | {circuit_data['transpiled_depth']} |

## Results
| Technique | Fidelity | Improvement vs Baseline | Total Shots | QPU Time |
|-----------|----------|------------------------|-------------|----------|
{rows}

**Winner**: {best['technique']} — fidelity **{best['fidelity']:.4f}** \
({((best['fidelity'] - baseline['fidelity']) / max(baseline['fidelity'], 0.001) * 100):+.1f}% vs baseline)

---
*Pipeline orchestrated by Prefect · IQM Garnet*
"""
    create_markdown_artifact(
        key="error-mitigation-report",
        markdown=report,
        description="Full error mitigation experiment report",
    )


# ═══════════════════════════════════════════════════════════════════════
# MAIN FLOW
# ═══════════════════════════════════════════════════════════════════════

@flow(
    name="GHZ Error Mitigation · IQM Garnet",
    description=(
        "Configurable error mitigation pipeline for GHZ state preparation. "
        "Toggle DD, REM, ZNE independently. Technique logic lives in "
        "dd.py, rem.py, zne.py — edit those files to customise."
    ),
    log_prints=True,
)
def ghz_mitigation_pipeline(
    num_qubits: int = 5,
    shots: int = 4096,
    enable_dd: bool = True,
    enable_rem: bool = True,
    enable_zne: bool = True,
    zne_scale_factors: list[int] = [1, 3, 5],
):
    active = [t for t, on in [("DD", enable_dd), ("REM", enable_rem), ("ZNE", enable_zne)] if on]

    print(f"\n{'━'*60}")
    print(f"  GHZ Error Mitigation · IQM Garnet")
    print(f"  Qubits: {num_qubits} | Shots: {shots}")
    print(f"  Techniques: {', '.join(active) if active else 'NONE (baseline only)'}")
    print(f"{'━'*60}\n")

    print("▸ STAGE 1: Build & Transpile")
    circuit_data = build_and_transpile(num_qubits)

    print("\n▸ STAGE 2: BASELINE")
    baseline = run_baseline(circuit_data, shots)
    all_results = [baseline]

    zne_result = rem_result = None

    if enable_dd:
        print("\n▸ STAGE 3a: Dynamical Decoupling")
        all_results.append(run_with_dd(circuit_data, shots))

    if enable_rem:
        print("\n▸ STAGE 3b: Readout Error Mitigation")
        rem_result = run_with_rem(circuit_data, shots)
        all_results.append(rem_result)

    if enable_zne:
        print(f"\n▸ STAGE 3c: Zero Noise Extrapolation (scales: {zne_scale_factors})")
        zne_result = run_with_zne(circuit_data, shots, zne_scale_factors)
        all_results.append(zne_result)

    if sum([enable_dd, enable_rem, enable_zne]) >= 2:
        print(f"\n▸ STAGE 3d: Combined ({'+'.join(active)})")
        all_results.append(run_combined(
            circuit_data, shots, enable_dd, enable_rem, enable_zne, zne_scale_factors,
        ))

    print("\n▸ STAGE 4: Generating Artifacts")
    publish_fidelity_chart(all_results)
    publish_heatmap(all_results, num_qubits)
    if zne_result:
        publish_zne_curve(zne_result)
    if rem_result:
        publish_rem_heatmap(rem_result, num_qubits)
    publish_report(all_results, circuit_data, enable_dd, enable_rem, enable_zne)

    best = max(all_results, key=lambda r: r["fidelity"])
    print(f"\n{'━'*60}")
    print(f"  ✅ Pipeline complete!")
    print(f"  Baseline fidelity : {baseline['fidelity']:.4f}")
    print(f"  Best fidelity     : {best['fidelity']:.4f} ({best['technique']})")
    print(f"  Improvement       : {((best['fidelity'] - baseline['fidelity']) / max(baseline['fidelity'], 0.001) * 100):+.1f}%")
    print(f"{'━'*60}\n")

    return {r["technique"]: r["fidelity"] for r in all_results}


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GHZ Error Mitigation · IQM Garnet")
    parser.add_argument("--qubits",     type=int,   default=5)
    parser.add_argument("--shots",      type=int,   default=4096)
    parser.add_argument("--no-dd",      action="store_true")
    parser.add_argument("--no-rem",     action="store_true")
    parser.add_argument("--no-zne",     action="store_true")
    parser.add_argument("--zne-scales", type=str,   default="1,3,5")
    args = parser.parse_args()

    ghz_mitigation_pipeline(
        num_qubits=args.qubits,
        shots=args.shots,
        enable_dd=not args.no_dd,
        enable_rem=not args.no_rem,
        enable_zne=not args.no_zne,
        zne_scale_factors=[int(s) for s in args.zne_scales.split(",")],
    )
