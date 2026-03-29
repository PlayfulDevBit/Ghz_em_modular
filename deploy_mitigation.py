"""
deploy_mitigation.py — Prefect 3.x Cloud Deployment
=====================================================
Registers the GHZ Error Mitigation pipeline to Prefect Cloud.

BEFORE RUNNING:
  1. Set WORK_POOL_NAME and GITHUB_URL below.
  2. Push all files to GitHub:
       git push origin main
     Required files in the repo root:
       ghz_mitigation_pipeline.py
       dd.py
       rem.py
       zne.py
  3. Ensure your IQM token is stored as a Prefect Secret:
       from prefect.blocks.system import Secret
       Secret(value="your-iqm-key").save("iqm-resonance-token")

Usage:
    python deploy_mitigation.py
"""

from prefect import flow
from prefect.runner.storage import GitRepository

# ── CONFIGURE THESE ─────────────────────────────────────────────────────
WORK_POOL_NAME = "my-managed-pool"                               # ← your work pool
GITHUB_URL     = "https://github.com/your-org/ghz-error-mitigation.git"  # ← your repo
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    deployment = flow.from_source(
        source=GitRepository(
            url=GITHUB_URL,
            branch="main",
        ),
        entrypoint="pipeline.py:ghz_mitigation_pipeline",
    )

    deployment_id = deployment.deploy(
        name="ghz-error-mitigation",
        work_pool_name=WORK_POOL_NAME,
        version="1.0.0",
        description=(
            "GHZ Error Mitigation on IQM Garnet. "
            "Toggleable DD, REM, and ZNE. "
            "Technique logic lives in dd.py, rem.py, zne.py — "
            "edit those files and push to update without touching the pipeline."
        ),
        tags=["quantum", "iqm-garnet", "error-mitigation", "ghz", "dd", "rem", "zne"],

        # ── Default parameters (all overridable from Prefect Cloud UI) ──
        parameters={
            "num_qubits":        5,
            "shots":             4096,
            "enable_dd":         True,
            "enable_rem":        True,
            "enable_zne":        True,
            "zne_scale_factors": [1, 3, 5],
        },

        job_variables={
            "pip_packages": [
                "qiskit==2.1.2",
                "iqm-client[qiskit]==33.0.5",
                "numpy>=1.24",
            ],
        },
    )

    print(f"\nDeployment registered: {deployment_id}")
    print("\nParameter toggles available in Prefect Cloud UI:")
    print("  num_qubits        int        — GHZ state size (default 5)")
    print("  shots             int        — QPU shots per run (default 4096)")
    print("  enable_dd         bool       — Dynamical Decoupling")
    print("  enable_rem        bool       — Readout Error Mitigation")
    print("  enable_zne        bool       — Zero Noise Extrapolation")
    print("  zne_scale_factors list[int]  — Noise scale factors, e.g. [1,3,5]")
    print("\nPlugin files pulled from GitHub at runtime:")
    print("  dd.py   — edit apply_dd()")
    print("  rem.py  — edit calibrate_rem() / apply_rem()")
    print("  zne.py  — edit fold_circuit() / extrapolate()")
    print("\nPrerequisites:")
    print("  - Prefect Cloud account with PREFECT_API_URL and PREFECT_API_KEY set")
    print("  - Prefect Secret block 'iqm-resonance-token' containing IQM Resonance API key")
    print(f"  - All files pushed to: {GITHUB_URL}")
    print("  - pipeline.py, dd.py, rem.py, zne.py all present in repo root")