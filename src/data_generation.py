import numpy as np
import pandas as pd
from pathlib import Path

# Path to save the synthetic dataset
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "synthetic_pupillometry.csv"

RNG = np.random.default_rng(42)

DIAGNOSES = ["TBI", "Stroke", "Sepsis", "Post-op", "Other"]
SITES = ["Site_A", "Site_B", "Site_C", "Site_D"]


def generate_synthetic_pupillometry(n_rows: int = 5000) -> pd.DataFrame:
    """Generate clinically realistic synthetic pupillometry data with noise,
    site-level bias, diagnosis effects, and overlapping distributions."""

    # -------------------------
    # Basic demographics
    # -------------------------
    patient_ids = np.arange(1, n_rows + 1)
    site_ids = RNG.choice(SITES, size=n_rows)
    ages = RNG.integers(18, 90, size=n_rows)
    sexes = RNG.choice(["M", "F"], size=n_rows)
    diagnoses = RNG.choice(DIAGNOSES, size=n_rows)

    # -------------------------
    # Base NPi distribution
    # -------------------------
    npi = np.clip(RNG.normal(3.2, 1.0, size=n_rows), 0, 5)

    # -------------------------
    # Add site-level bias (real hospitals differ)
    # -------------------------
    site_bias = {
        "Site_A": RNG.normal(0, 0.4),
        "Site_B": RNG.normal(0, 0.6),
        "Site_C": RNG.normal(0, 0.3),
        "Site_D": RNG.normal(0, 0.8),
    }
    npi += np.array([site_bias[s] for s in site_ids])
    npi = np.clip(npi, 0, 5)

    # -------------------------
    # Diagnosis-specific effects
    # -------------------------
    diag_effect = {
        "TBI": -0.4,
        "Stroke": -0.2,
        "Sepsis": -0.1,
        "Post-op": +0.1,
        "Other": 0.0,
    }
    npi += np.array([diag_effect[d] for d in diagnoses])
    npi = np.clip(npi, 0, 5)

    # -------------------------
    # Add outliers (blink artifacts, bad captures)
    # -------------------------
    outlier_idx = RNG.choice(n_rows, size=int(n_rows * 0.03), replace=False)
    npi[outlier_idx] = np.clip(
        npi[outlier_idx] + RNG.normal(0, 2.0, size=len(outlier_idx)),
        0, 5
    )

    # -------------------------
    # GCS with weaker correlation + more noise
    # -------------------------
    gcs_noise = RNG.normal(0, 3.5, size=n_rows)
    gcs = np.clip((npi * 2.0) + 7 + gcs_noise, 3, 15).round().astype(int)

    # Binary severe flag
    gcs_severe = (gcs <= 8).astype(int)

    # Severity label
    severity = np.where(gcs <= 8, "Severe",
                 np.where(gcs <= 12, "Moderate", "Mild"))

    # -------------------------
    # True pupil size
    # -------------------------
    true_pupil_size = np.clip(RNG.normal(3.0, 0.7, size=n_rows), 1.0, 7.0)

    # -------------------------
    # Measurement noise (more realistic)
    # -------------------------
    measured_pupil_size = np.clip(
        true_pupil_size + RNG.normal(0, 0.5, size=n_rows),
        1.0, 7.0
    )

    # Left/right pupils with jitter
    pupil_left = np.clip(
        measured_pupil_size + RNG.normal(0, 0.3, size=n_rows),
        1.0, 7.0
    )
    pupil_right = np.clip(
        measured_pupil_size + RNG.normal(0, 0.3, size=n_rows),
        1.0, 7.0
    )

    # -------------------------
    # Pupil dynamics (more overlap)
    # -------------------------
    constriction_velocity = np.clip(
        RNG.normal(1.5, 0.6, size=n_rows) - (gcs_severe * 0.3),
        0.1, None
    )
    dilation_velocity = np.clip(
        RNG.normal(1.2, 0.5, size=n_rows) - (gcs_severe * 0.25),
        0.1, None
    )
    latency_ms = np.clip(
        RNG.normal(250, 50, size=n_rows) + (gcs_severe * 60),
        150, 600
    ).astype(int)

    # -------------------------
    # Build DataFrame
    # -------------------------
    df = pd.DataFrame({
        "patient_id": patient_ids,
        "site_id": site_ids,
        "age": ages,
        "sex": sexes,
        "diagnosis": diagnoses,
        "npi": npi.round(2),
        "true_pupil_size": true_pupil_size.round(2),
        "measured_pupil_size": measured_pupil_size.round(2),
        "pupil_left": pupil_left.round(2),
        "pupil_right": pupil_right.round(2),
        "constriction_velocity": constriction_velocity.round(2),
        "dilation_velocity": dilation_velocity.round(2),
        "latency_ms": latency_ms,
        "gcs": gcs,
        "gcs_severe": gcs_severe,
        "severity": severity
    })

    return df


def save_synthetic_data(path: Path = DATA_PATH, n_rows: int = 5000):
    df = generate_synthetic_pupillometry(n_rows)
    df.to_csv(path, index=False)
    print(f"Synthetic dataset saved to: {path}")


if __name__ == "__main__":
    save_synthetic_data()