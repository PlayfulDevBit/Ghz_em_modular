# GHZ Error Mitigation Pipeline · IQM Garnet

Run a GHZ state preparation experiment on IQM Garnet with toggleable error
reduction techniques. Results are published as SVG charts and a comparison
report to the Prefect dashboard.

---

## Quick-start glossary

| Term | What it means here |
|------|--------------------|
| **GHZ state** | A maximally entangled quantum state: (|000…0⟩ + |111…1⟩)/√2. A useful benchmark because any error shows up as measurements in the "wrong" bitstrings. |
| **Fidelity** | Fraction of measurements that landed in the two correct GHZ bitstrings (|000…0⟩ or |111…1⟩). Perfect = 1.0, random noise = ~0. |
| **DD** | Dynamical Decoupling — suppresses idle-qubit noise by inserting cancelling pulse pairs. |
| **REM** | Readout Error Mitigation — corrects measurement errors using calibration circuits. |
| **ZNE** | Zero Noise Extrapolation — runs the circuit at multiple noise levels and extrapolates to zero noise. |
| **Prefect** | The workflow orchestrator that schedules, logs, and retries the pipeline tasks. |
| **IQM Garnet** | The 20-qubit superconducting QPU the pipeline targets. |

---

## File structure

```
ghz-error-mitigation/
│
├── dd.py                        ← Dynamical Decoupling plugin (edit me)
├── rem.py                       ← Readout Error Mitigation plugin (edit me)
├── zne.py                       ← Zero Noise Extrapolation plugin (edit me)
│
├── ghz_mitigation_pipeline.py   ← Main Prefect flow (imports the plugins)
├── deploy_mitigation.py         ← Registers the flow to Prefect Cloud
└── README.md                    ← This file
```

**Rule of thumb:** you only ever need to edit the three plugin files.
The pipeline and deploy scripts never need to change unless you want to
add a completely new technique.

---

## How the pipeline works

```
STAGE 1   Build & transpile GHZ-N circuit
    ↓
STAGE 2   Run baseline (no mitigation) → fidelity_baseline
    ↓
STAGE 3a  [if DD enabled]  apply_dd()  → run on QPU → fidelity_dd
STAGE 3b  [if REM enabled] calibrate_rem() → run → apply_rem() → fidelity_rem
STAGE 3c  [if ZNE enabled] fold_circuit() × scales → extrapolate() → fidelity_zne
STAGE 3d  [if 2+ enabled]  all techniques combined → fidelity_combined
    ↓
STAGE 4   Publish SVG artifacts + report to Prefect dashboard
```

Each stage in 3a–3d is independent — they run in whatever order Prefect
schedules them and compare against the same baseline.

---

## How to customise a technique

The workflow is the same for DD, REM, and ZNE:

### Step 1 — Download the plugin file

```bash
# clone the repo, or just download the single file
git clone https://github.com/your-org/ghz-error-mitigation.git
cd ghz-error-mitigation
```

### Step 2 — Open in Jupyter and edit

```python
# In a Jupyter cell, open the file inline:
%load rem.py
```

Or open `rem.py` / `dd.py` / `zne.py` directly in JupyterLab.

Each file contains:
- A short explanation of what the technique does.
- The function(s) you can customise, with clearly marked sections.
- A "CUSTOMISATION IDEAS" comment block with concrete suggestions.

### Step 3 — Run the local test

Every plugin file has a built-in test at the bottom that runs without a
QPU token (it uses a fake backend or synthetic data):

```bash
python rem.py    # tests calibrate_rem + apply_rem with a fake backend
python dd.py     # tests apply_dd (graceful fallback without real timing data)
python zne.py    # tests fold_circuit + extrapolate with simulated fidelities
```

If the script ends with `✅ ... test passed`, your changes are safe to upload.

### Step 4 — Run a small local experiment (optional, needs QPU token)

You can call the plugin functions directly in a notebook before running the
full pipeline. This costs only a few QPU shots.

**REM example:**
```python
from rem import calibrate_rem, apply_rem
from qiskit import QuantumCircuit, transpile
from iqm.qiskit_iqm import IQMProvider

token   = "your-iqm-resonance-token"
backend = IQMProvider("https://cocos.resonance.meetiqm.com/garnet", token=token).get_backend()

# Run calibration
cal = calibrate_rem(backend, n_qubits=3, shots=1024)
print("Readout errors:", cal["qubit_readout_errors"])

# Build a small test circuit
qc = QuantumCircuit(3)
qc.h(0); qc.cx(0, 1); qc.cx(0, 2); qc.measure_all()
qc_t = transpile(qc, backend=backend, optimization_level=2)

# Run and correct
raw = backend.run(qc_t, shots=1024, use_timeslot=False).result().get_counts()
corrected = apply_rem(raw, cal, n_qubits=3, shots=1024)
print("Raw:      ", raw)
print("Corrected:", corrected)
```

**ZNE example:**
```python
from zne import fold_circuit, extrapolate

fidelities = []
for scale in [1, 3, 5]:
    qc_folded = fold_circuit(qc_t, scale)
    counts = backend.run(qc_folded, shots=1024, use_timeslot=False).result().get_counts()
    ghz_fid = (counts.get("000", 0) + counts.get("111", 0)) / 1024
    fidelities.append(ghz_fid)
    print(f"scale={scale}: fidelity={ghz_fid:.4f}")

print("ZNE estimate:", extrapolate([1, 3, 5], fidelities))
```

**DD example:**
```python
from dd import apply_dd

qc_t   = transpile(qc, backend=backend, optimization_level=2)
qc_dd  = apply_dd(qc_t, backend)
counts = backend.run(qc_dd, shots=1024, use_timeslot=False).result().get_counts()
```

### Step 5 — Upload and run the full pipeline

Once your local test passes, push the modified plugin file:

```bash
git add rem.py          # or dd.py / zne.py
git commit -m "try NNLS correction in REM"
git push origin main
```

Then trigger the pipeline from Prefect Cloud — it will pull the latest
code from GitHub automatically. No changes to the pipeline file needed.

---

## Interface contract — rules you must not break

The pipeline calls specific function names with specific signatures.
As long as you keep these, you can change anything inside the functions.

| File | Functions the pipeline calls | Signature (must stay exactly this) |
|------|-----------------------------|------------------------------------|
| `rem.py` | `calibrate_rem` | `(backend, n_qubits: int, shots: int) → CalibrationData` |
| `rem.py` | `apply_rem` | `(raw_counts: dict, calibration: CalibrationData, n_qubits: int, shots: int) → dict[str, int]` |
| `dd.py` | `apply_dd` | `(circuit, backend) → QuantumCircuit` |
| `zne.py` | `fold_circuit` | `(circuit, scale_factor: int) → QuantumCircuit` |
| `zne.py` | `extrapolate` | `(scale_factors: list[int], fidelities: list[float]) → float` |

**Other rules:**
- `apply_rem` must return a dict of `{bitstring: int_count}` — the pipeline computes fidelity from it.
- `extrapolate` must return a float between 0.0 and 1.0.
- `apply_dd` must return a QuantumCircuit. On failure, return the input circuit unchanged (don't raise).
- `fold_circuit` must accept `scale_factor=1` as a no-op (returns circuit unchanged).
- Do not add required imports outside the standard library / Qiskit / numpy at the top level — the pipeline environment installs only what is listed in `deploy_mitigation.py`.

---

## Running the full pipeline

### Local (no Prefect Cloud needed)

```bash
# All techniques enabled
python ghz_mitigation_pipeline.py

# REM only
python ghz_mitigation_pipeline.py --no-dd --no-zne

# Custom qubits and shots
python ghz_mitigation_pipeline.py --qubits 3 --shots 2048 --zne-scales 1,3,5,7
```

Set your IQM token as an environment variable or a Prefect Secret block:

```bash
# Option A — environment variable (local use only)
export IQM_TOKEN="your-token-here"

# Option B — Prefect Secret (works locally and in Cloud)
python -c "
from prefect.blocks.system import Secret
Secret(value='your-token-here').save('iqm-resonance-token')
"
```

### Prefect Cloud (serverless, triggered from UI)

```bash
# 1. Make sure PREFECT_API_URL and PREFECT_API_KEY are set
export PREFECT_API_URL="https://api.prefect.cloud/api/accounts/.../workspaces/..."
export PREFECT_API_KEY="your-prefect-key"

# 2. Push your code to GitHub
git push origin main

# 3. Register the deployment (one-time, or after changing deploy_mitigation.py)
python deploy_mitigation.py

# 4. Trigger a run from the Prefect Cloud UI
#    → Deployments → ghz-error-mitigation → Run
#    Toggle enable_dd / enable_rem / enable_zne in the parameters panel
```

Artifacts (fidelity chart, heatmap, ZNE curve, REM error map, report)
appear in the Prefect dashboard under **Artifacts** after each run.

---

## Dependencies

```
qiskit==2.1.2
iqm-client[qiskit]==33.0.5
numpy>=1.24
prefect>=3.0
```

Install locally:
```bash
pip install qiskit==2.1.2 "iqm-client[qiskit]==33.0.5" numpy prefect
```
