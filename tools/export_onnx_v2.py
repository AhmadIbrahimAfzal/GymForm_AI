"""
Export gym_model_v2.pt to ONNX with parity verification.
=========================================================
This exports the IMPROVED v2 model (8->128->64->32->10 with BatchNorm),
NOT the old model (8->64->32->10).

Also bundles scaler_params.json alongside the ONNX model so the mobile
app can normalize inputs before inference.

Run: python tools/export_onnx_v2.py
Outputs:
  - mobile/assets/models/gym_model_v2.onnx
  - mobile/assets/models/scaler_params.json (copied from root)

Original export_onnx.py from the Phase 1 plan is NOT modified.
"""
import sys
import json
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
ONNX_OUT = ROOT / "mobile" / "assets" / "models" / "gym_model_v2.onnx"
SCALER_SRC = ROOT / "scaler_params.json"
SCALER_DST = ONNX_OUT.parent / "scaler_params.json"
ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)


class GymModelV2(nn.Module):
    """
    Must match the architecture in train_model_v2.py EXACTLY.
    8 -> 128 (BN, LeakyReLU, Drop) -> 64 (BN, LeakyReLU, Drop) -> 32 (BN, LeakyReLU) -> 10
    """
    def __init__(self, num_features=8, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),

            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def main():
    # Load the v2 model
    model_path = ROOT / "gym_model_v2.pt"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found. Run train_model_v2.py first.")
        sys.exit(1)

    model = GymModelV2()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # CRITICAL: switches BatchNorm to inference mode

    # Export to ONNX
    dummy_input = torch.randn(1, 8)
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_OUT),
        input_names=["angles"],       # normalized angles, shape [1, 8]
        output_names=["logits"],      # raw logits, shape [1, 10]
        opset_version=17,
        dynamic_axes={"angles": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"[OK] Exported ONNX model to {ONNX_OUT}")
    print(f"     Size: {ONNX_OUT.stat().st_size / 1024:.1f} KB")

    # Copy scaler params alongside the model
    if not SCALER_SRC.exists():
        print(f"ERROR: {SCALER_SRC} not found. Run train_model_v2.py first.")
        sys.exit(1)
    shutil.copy2(SCALER_SRC, SCALER_DST)
    print(f"[OK] Copied scaler_params.json to {SCALER_DST}")

    # ── Parity check: ONNX vs PyTorch ──
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARN] onnxruntime not installed. Skipping parity check.")
        print("       Run: pip install onnxruntime")
        return

    sess = ort.InferenceSession(str(ONNX_OUT))

    # Load scaler params to normalize test inputs the same way
    with open(SCALER_SRC) as f:
        scaler = json.load(f)
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)

    # Generate random angle-like inputs, normalize them
    rng = np.random.default_rng(42)
    raw_angles = rng.uniform(20, 180, size=(50, 8)).astype(np.float32)
    normalized = (raw_angles - mean) / std

    # Compare PyTorch vs ONNX on normalized inputs
    with torch.no_grad():
        torch_out = model(torch.from_numpy(normalized)).numpy()

    onnx_out = np.concatenate([
        sess.run(None, {"angles": normalized[i:i+1]})[0]
        for i in range(len(normalized))
    ])

    max_diff = float(np.abs(torch_out - onnx_out).max())
    assert max_diff < 1e-4, f"ONNX diverges from PyTorch: max diff {max_diff}"
    print(f"[OK] Parity check passed: max diff = {max_diff:.2e}")

    # Also verify the scaler params are correct
    print(f"\n[INFO] Scaler params (the app MUST normalize inputs with these):")
    for i, name in enumerate(scaler["feature_names"]):
        print(f"  {name:>12}: mean={scaler['mean'][i]:.2f}, std={scaler['std'][i]:.2f}")

    print(f"\n[OK] Ready for mobile. Files:")
    print(f"     Model:  {ONNX_OUT}")
    print(f"     Scaler: {SCALER_DST}")
    print(f"\n     IMPORTANT: The mobile classifier.ts MUST normalize raw angles")
    print(f"     using scaler_params.json BEFORE feeding them to the ONNX model.")
    print(f"     Formula: normalized_angle = (raw_angle - mean) / std")


if __name__ == "__main__":
    main()
