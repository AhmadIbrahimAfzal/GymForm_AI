"""Generate JSON parity fixtures from the Python reference implementation.

Uses the V2 model with normalization for classifier fixtures.
All other fixtures (angles, EMA, majority vote, exercises) are
identical to the original — they test pure logic, not the model.
"""
import json, math, random, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn as nn
from exercises import BicepCurl, Squat, LateralRaise, ShoulderPress, TricepFinisher

OUT = Path(__file__).resolve().parents[1] / "mobile" / "src" / "core" / "__fixtures__"
OUT.mkdir(parents=True, exist_ok=True)
rng = random.Random(42)

def calculate_angle(a, b, c):  # verbatim from app.py
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return float(360 - angle if angle > 180 else angle)

# --- angles.json: [{a, b, c, expected}] ---
angle_cases = []
for _ in range(200):
    pts = [[rng.uniform(0, 1), rng.uniform(0, 1)] for _ in range(3)]
    angle_cases.append({"a": pts[0], "b": pts[1], "c": pts[2],
                        "expected": calculate_angle(*pts)})
(OUT / "angles.json").write_text(json.dumps(angle_cases))
print(f"  angles.json: {len(angle_cases)} cases")

# --- ema.json: {alpha, frames: [[[x,y,z]x33]xN], expected: same shape} ---
ALPHA = 0.5
frames = [[[rng.uniform(0, 1) for _ in range(3)] for _ in range(33)] for _ in range(30)]
state, expected = None, []
for f in frames:
    if state is None:
        state = [list(lm) for lm in f]
    else:
        for i, lm in enumerate(f):
            for k in range(3):
                state[i][k] = ALPHA * lm[k] + (1 - ALPHA) * state[i][k]
    expected.append([list(lm) for lm in state])
(OUT / "ema.json").write_text(json.dumps({"alpha": ALPHA, "frames": frames, "expected": expected}))
print(f"  ema.json: {len(frames)} frames")

# --- majority_vote.json: {maxlen, sequence: [label], expected: [label]} ---
LABELS = ["Good Curl", "Bad Curl", "Good Squat", "Bad Squat"]
seq = [rng.choice(LABELS) for _ in range(60)]
hist, mv_expected = [], []
for label in seq:
    hist.append(label)
    if len(hist) > 10:
        hist.pop(0)
    mv_expected.append(Counter(hist).most_common(1)[0][0])
(OUT / "majority_vote.json").write_text(json.dumps({"maxlen": 10, "sequence": seq, "expected": mv_expected}))
print(f"  majority_vote.json: {len(seq)} labels")

# --- exercises.json: per exercise, a step sequence with expected (reps, stage) ---
def run_machine(machine, steps):
    out = []
    for angles, label in steps:
        reps, stage = machine.update(angles, label)
        out.append({"angles": angles, "label": label, "reps": reps, "stage": stage})
    return out

def curl_steps():
    s = []
    for _ in range(3):  # 3 good reps
        s += [({"l_elbow": 160.0, "r_elbow": 160.0}, "Good Curl"),
              ({"l_elbow": 60.0, "r_elbow": 60.0}, "Good Curl")]
    s += [({"l_elbow": 160.0, "r_elbow": 160.0}, "Good Curl"),
          ({"l_elbow": 100.0, "r_elbow": 100.0}, "Bad Curl"),   # bad mid-rep
          ({"l_elbow": 60.0, "r_elbow": 60.0}, "Good Curl")]    # must NOT count
    return s

def squat_steps():
    s = []
    for _ in range(2):
        s += [({"l_knee": 170.0, "r_knee": 170.0}, "Good Squat"),
              ({"l_knee": 100.0, "r_knee": 100.0}, "Good Squat")]
    s += [({"l_knee": 170.0, "r_knee": 170.0}, "Good Squat"),
          ({"l_knee": 100.0, "r_knee": 100.0}, "Bad Squat")]
    return s

def raise_steps():
    return [({"l_shoulder": 20.0, "r_shoulder": 20.0}, "Good Raise"),
            ({"l_shoulder": 80.0, "r_shoulder": 80.0}, "Good Raise"),
            ({"l_shoulder": 20.0, "r_shoulder": 20.0}, "Good Raise"),
            ({"l_shoulder": 80.0, "r_shoulder": 80.0}, "Bad Raise"),
            ({"l_shoulder": 20.0, "r_shoulder": 20.0}, "Good Raise")]

def press_steps():
    return [({"l_elbow": 80.0, "r_elbow": 80.0}, "Good Shoulder"),
            ({"l_elbow": 170.0, "r_elbow": 170.0}, "Good Shoulder"),
            ({"l_elbow": 80.0, "r_elbow": 80.0}, "Good Shoulder"),
            ({"l_elbow": 170.0, "r_elbow": 170.0}, "Good Shoulder")]

def tricep_steps():
    return [({"l_elbow": 50.0, "active_arm": "left"}, "Good Tricep"),
            ({"l_elbow": 170.0, "active_arm": "left"}, "Good Tricep"),
            ({"r_elbow": 50.0, "active_arm": "right"}, "Good Tricep"),
            ({"r_elbow": 170.0, "active_arm": "right"}, "Bad Tricep"),
            ({"r_elbow": 50.0, "active_arm": "right"}, "Good Tricep")]

exercises_fixture = {
    "Bicep Curl": run_machine(BicepCurl(), curl_steps()),
    "Squat": run_machine(Squat(), squat_steps()),
    "Lateral Raise": run_machine(LateralRaise(), raise_steps()),
    "Shoulder Press": run_machine(ShoulderPress(), press_steps()),
    "Tricep Finisher": run_machine(TricepFinisher(), tricep_steps()),
}
(OUT / "exercises.json").write_text(json.dumps(exercises_fixture))
print(f"  exercises.json: {len(exercises_fixture)} exercises")

# --- classifier.json: rows of 8 angles -> v2 model logits + per-exercise verdicts ---
# Using the V2 model with StandardScaler normalization
class GymModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.1),
            nn.Linear(32, 10))
    def forward(self, x):
        return self.network(x)

LABELS_MAP_REVERSE = {0: "Bad Curl", 1: "Good Curl", 2: "Bad Squat", 3: "Good Squat",
                      4: "Bad Raise", 5: "Good Raise", 6: "Bad Shoulder", 7: "Good Shoulder",
                      8: "Bad Tricep", 9: "Good Tricep"}
MASK_RANGES = {"Bicep Curl": (0, 2), "Squat": (2, 4), "Lateral Raise": (4, 6),
               "Shoulder Press": (6, 8), "Tricep Finisher": (8, 10)}

ROOT = Path(__file__).resolve().parents[1]
model = GymModelV2()
model.load_state_dict(torch.load(ROOT / "gym_model_v2.pt", weights_only=True))
model.eval()

# Load scaler for normalization
with open(ROOT / "scaler_params.json") as f:
    scaler = json.load(f)
scaler_mean = np.array(scaler["mean"], dtype=np.float32)
scaler_std = np.array(scaler["std"], dtype=np.float32)

UPPER_BODY = ["Bicep Curl", "Lateral Raise", "Shoulder Press", "Tricep Finisher"]

rows = []
for _ in range(50):
    raw_angles = [rng.uniform(20, 180) for _ in range(8)]
    # Standard raw logits (unmasked angles)
    normalized_raw = (np.array(raw_angles, dtype=np.float32) - scaler_mean) / scaler_std
    with torch.no_grad():
        raw_logits = model(torch.FloatTensor([normalized_raw]))[0]

    verdicts = {}
    for ex, (lo, hi) in MASK_RANGES.items():
        # Mask irrelevant angles to 180 (matching app.py and anglesToInput)
        ex_angles = list(raw_angles)
        if ex in UPPER_BODY:
            ex_angles[4] = ex_angles[5] = ex_angles[6] = ex_angles[7] = 180.0
        else:
            ex_angles[0] = ex_angles[1] = ex_angles[2] = ex_angles[3] = 180.0

        normalized_ex = (np.array(ex_angles, dtype=np.float32) - scaler_mean) / scaler_std
        with torch.no_grad():
            ex_logits = model(torch.FloatTensor([normalized_ex]))[0]

        masked = ex_logits.clone()
        masked[:lo] = float("-inf")
        masked[hi:] = float("-inf")
        probs = torch.softmax(masked, dim=0)
        idx = int(torch.argmax(probs))
        verdicts[ex] = {"label": LABELS_MAP_REVERSE[idx],
                        "confidence": float(probs[idx]) * 100.0,
                        "logits": [float(v) for v in ex_logits]}
    rows.append({"angles": raw_angles, "logits": [float(v) for v in raw_logits], "verdicts": verdicts})
(OUT / "classifier.json").write_text(json.dumps(rows))
print(f"  classifier.json: {len(rows)} rows (using GymModelV2 + normalization)")

print(f"\nWrote all fixtures to {OUT}")
