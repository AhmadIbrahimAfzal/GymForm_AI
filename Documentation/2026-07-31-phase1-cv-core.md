# Phase 1 — CV Core (Live Coach) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A React Native app that runs the GymForm AI live coach fully on-device: camera → MediaPipe pose → ONNX form classifier → strict rep counting, with skeleton overlay, guided session flow, and voice cues.

**Architecture:** The existing Python pipeline is ported piecewise. Heavy CV (pose landmarking) runs in a native VisionCamera frame-processor plugin (MediaPipe Tasks, Kotlin/Swift). All decision logic (angles, smoothing, classification masking, rep state machines) is pure TypeScript in `mobile/src/core/`, developed test-first against JSON fixtures generated from the Python reference so behavior provably matches. The classifier is the PyTorch MLP exported once to ONNX. UI is Expo (dev-client) + Skia overlay.

**Tech Stack:** Expo SDK (dev-client builds), TypeScript (strict), react-native-vision-camera v4, MediaPipe Tasks Vision (Android/iOS SDKs), onnxruntime-react-native (app) / onnxruntime-node (tests), @shopify/react-native-skia, react-native-reanimated, expo-speech, Jest.

## Global Constraints

- The Python code in the repo root (`app.py`, `exercises.py`) is the **behavioral reference**. Fixtures generated from it are the source of truth; TS ports must match within 1e-4 (floats) / exactly (labels, rep counts, stages).
- No video or landmarks ever leave the device. No network calls in Phase 1.
- All new app code lives under `mobile/`. Python tooling scripts live under `tools/`.
- TypeScript `strict: true`. No `any` in `mobile/src/core/`.
- `mobile/src/core/` must stay dependency-free (no React, no native imports) — pure functions/classes only.
- Camera target: 480×360 (matches current app), 30 fps request, 15 fps floor.
- Model classes (index → label): 0 Bad Curl, 1 Good Curl, 2 Bad Squat, 3 Good Squat, 4 Bad Raise, 5 Good Raise, 6 Bad Shoulder, 7 Good Shoulder, 8 Bad Tricep, 9 Good Tricep.
- Exercise names (exact strings): `"Bicep Curl"`, `"Squat"`, `"Lateral Raise"`, `"Shoulder Press"`, `"Tricep Finisher"`.
- Commit after every task (at minimum); use conventional-commit style messages.

---

### Task 1: Scaffold the Expo app

**Files:**
- Create: `mobile/` (via `create-expo-app`), `mobile/tsconfig.json`, `mobile/jest.config.js`
- Create: `mobile/src/core/__tests__/smoke.test.ts`
- Modify: `.gitignore` (append mobile ignores)

**Interfaces:**
- Produces: a bootable Expo TypeScript app; `npm test` runs Jest from `mobile/`.

- [ ] **Step 1: Scaffold**

```bash
cd /Users/admin/Projects/ReactNative/GymForm_AI
npx create-expo-app@latest mobile --template blank-typescript
cd mobile
npx expo install expo-dev-client
npm install --save-dev jest ts-jest @types/jest
```

- [ ] **Step 2: Configure strict TS and Jest**

`mobile/tsconfig.json` — ensure:

```json
{
  "extends": "expo/tsconfig.base",
  "compilerOptions": { "strict": true }
}
```

`mobile/jest.config.js`:

```js
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src'],
};
```

Add to `mobile/package.json` scripts: `"test": "jest"`.

- [ ] **Step 3: Smoke test**

`mobile/src/core/__tests__/smoke.test.ts`:

```ts
test('jest runs', () => {
  expect(1 + 1).toBe(2);
});
```

Run: `cd mobile && npm test` — Expected: PASS.

- [ ] **Step 4: Verify the app boots**

Run: `cd mobile && npx expo start` and open in a simulator or Expo Go. Expected: default blank screen renders without errors.

- [ ] **Step 5: Ignore artifacts and commit**

Append to repo-root `.gitignore`:

```
mobile/node_modules/
mobile/.expo/
mobile/ios/build/
mobile/android/build/
mobile/android/app/build/
```

```bash
git add .gitignore mobile
git commit -m "feat(mobile): scaffold Expo TypeScript app with Jest"
```

---

### Task 2: Generate parity fixtures from the Python reference

**Files:**
- Create: `tools/export_fixtures.py`
- Create (generated): `mobile/src/core/__fixtures__/angles.json`, `ema.json`, `majority_vote.json`, `exercises.json`, `classifier.json`

**Interfaces:**
- Consumes: root `exercises.py`, `dataset_fullbody.csv`, `gym_model_fullbody.pt`, the `calculate_angle` / EMA / majority-vote logic from `app.py` (reimplemented inline here to avoid importing Streamlit).
- Produces: five JSON fixture files. Shapes documented in Step 1 — later tasks' tests consume exactly these.

- [ ] **Step 1: Write the export script**

`tools/export_fixtures.py`:

```python
"""Generate JSON parity fixtures from the Python reference implementation."""
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

# --- classifier.json: rows of 8 angles -> torch logits + per-exercise verdicts ---
class GymModel(nn.Module):  # verbatim from train_model.py
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    def forward(self, x):
        return self.network(x)

LABELS_MAP_REVERSE = {0: "Bad Curl", 1: "Good Curl", 2: "Bad Squat", 3: "Good Squat",
                      4: "Bad Raise", 5: "Good Raise", 6: "Bad Shoulder", 7: "Good Shoulder",
                      8: "Bad Tricep", 9: "Good Tricep"}
MASK_RANGES = {"Bicep Curl": (0, 2), "Squat": (2, 4), "Lateral Raise": (4, 6),
               "Shoulder Press": (6, 8), "Tricep Finisher": (8, 10)}

model = GymModel()
model.load_state_dict(torch.load(Path(__file__).resolve().parents[1] / "gym_model_fullbody.pt",
                                 weights_only=True))
model.eval()

rows = []
for _ in range(50):
    angles = [rng.uniform(20, 180) for _ in range(8)]
    with torch.no_grad():
        logits = model(torch.FloatTensor([angles]))[0]
    verdicts = {}
    for ex, (lo, hi) in MASK_RANGES.items():
        masked = logits.clone()
        masked[:lo] = float("-inf")
        masked[hi:] = float("-inf")
        probs = torch.softmax(masked, dim=0)
        idx = int(torch.argmax(probs))
        verdicts[ex] = {"label": LABELS_MAP_REVERSE[idx],
                        "confidence": float(probs[idx]) * 100.0}
    rows.append({"angles": angles, "logits": [float(v) for v in logits], "verdicts": verdicts})
(OUT / "classifier.json").write_text(json.dumps(rows))

print(f"Wrote fixtures to {OUT}")
```

- [ ] **Step 2: Run it**

Run: `python tools/export_fixtures.py`
Expected: `Wrote fixtures to .../mobile/src/core/__fixtures__` and five JSON files exist. Spot-check `exercises.json`: `"Bicep Curl"` sequence ends with `reps: 3` (the 4th rep had bad form mid-rep and must not count).

- [ ] **Step 3: Commit**

```bash
git add tools/export_fixtures.py mobile/src/core/__fixtures__
git commit -m "feat(tools): export parity fixtures from Python reference"
```

---

### Task 3: Core — angle math

**Files:**
- Create: `mobile/src/core/angles.ts`
- Test: `mobile/src/core/__tests__/angles.test.ts`

**Interfaces:**
- Produces:
  - `type Point = [number, number]`
  - `calculateAngle(a: Point, b: Point, c: Point): number`
  - `type JointAngles = { l_elbow: number; r_elbow: number; l_shoulder: number; r_shoulder: number; l_hip: number; r_hip: number; l_knee: number; r_knee: number }`
  - `computeJointAngles(landmarks: number[][]): JointAngles` — takes 33 landmarks `[x, y, z]`, uses indices 11–16 and 23–28 exactly as `app.py` does.

- [ ] **Step 1: Failing test**

`mobile/src/core/__tests__/angles.test.ts`:

```ts
import { calculateAngle, computeJointAngles, Point } from '../angles';
import cases from '../__fixtures__/angles.json';

test('calculateAngle matches Python reference', () => {
  for (const c of cases as { a: Point; b: Point; c: Point; expected: number }[]) {
    expect(calculateAngle(c.a, c.b, c.c)).toBeCloseTo(c.expected, 4);
  }
});

test('computeJointAngles uses the correct landmark indices', () => {
  const lms = Array.from({ length: 33 }, (_, i) => [i / 100, i / 200, 0]);
  const a = computeJointAngles(lms);
  expect(a.l_elbow).toBeCloseTo(
    calculateAngle([lms[11][0], lms[11][1]], [lms[13][0], lms[13][1]], [lms[15][0], lms[15][1]]), 6);
  expect(a.r_knee).toBeCloseTo(
    calculateAngle([lms[24][0], lms[24][1]], [lms[26][0], lms[26][1]], [lms[28][0], lms[28][1]]), 6);
});
```

Add to `jest.config.js` (so JSON imports work): `moduleFileExtensions: ['ts', 'js', 'json']` and in `tsconfig.json` compilerOptions: `"resolveJsonModule": true`.

- [ ] **Step 2: Run to verify failure**

Run: `cd mobile && npm test -- angles`
Expected: FAIL — cannot find module `../angles`.

- [ ] **Step 3: Implement**

`mobile/src/core/angles.ts`:

```ts
export type Point = [number, number];

export function calculateAngle(a: Point, b: Point, c: Point): number {
  const radians = Math.atan2(c[1] - b[1], c[0] - b[0]) - Math.atan2(a[1] - b[1], a[0] - b[0]);
  const angle = Math.abs((radians * 180.0) / Math.PI);
  return angle > 180 ? 360 - angle : angle;
}

export interface JointAngles {
  l_elbow: number; r_elbow: number;
  l_shoulder: number; r_shoulder: number;
  l_hip: number; r_hip: number;
  l_knee: number; r_knee: number;
}

const p = (lms: number[][], i: number): Point => [lms[i][0], lms[i][1]];

export function computeJointAngles(lms: number[][]): JointAngles {
  return {
    l_elbow: calculateAngle(p(lms, 11), p(lms, 13), p(lms, 15)),
    r_elbow: calculateAngle(p(lms, 12), p(lms, 14), p(lms, 16)),
    l_shoulder: calculateAngle(p(lms, 23), p(lms, 11), p(lms, 13)),
    r_shoulder: calculateAngle(p(lms, 24), p(lms, 12), p(lms, 14)),
    l_hip: calculateAngle(p(lms, 11), p(lms, 23), p(lms, 25)),
    r_hip: calculateAngle(p(lms, 12), p(lms, 24), p(lms, 26)),
    l_knee: calculateAngle(p(lms, 23), p(lms, 25), p(lms, 27)),
    r_knee: calculateAngle(p(lms, 24), p(lms, 26), p(lms, 28)),
  };
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd mobile && npm test -- angles` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mobile/src/core/angles.ts mobile/src/core/__tests__/angles.test.ts mobile/jest.config.js mobile/tsconfig.json
git commit -m "feat(core): port angle math with Python parity tests"
```

---

### Task 4: Core — EMA smoothing and majority vote

**Files:**
- Create: `mobile/src/core/ema.ts`, `mobile/src/core/majorityVote.ts`
- Test: `mobile/src/core/__tests__/smoothing.test.ts`

**Interfaces:**
- Produces:
  - `class EmaFilter { constructor(alpha?: number); apply(landmarks: number[][]): number[][]; reset(): void }` (default alpha 0.5; returns the smoothed copy, does not mutate input)
  - `class MajorityVote { constructor(maxlen?: number); add(label: string): string; reset(): void }` (default maxlen 10; tie-break = first label to reach the max count in history order, matching Python `Counter.most_common`)

- [ ] **Step 1: Failing test**

`mobile/src/core/__tests__/smoothing.test.ts`:

```ts
import { EmaFilter } from '../ema';
import { MajorityVote } from '../majorityVote';
import emaFx from '../__fixtures__/ema.json';
import mvFx from '../__fixtures__/majority_vote.json';

test('EmaFilter matches Python reference', () => {
  const f = new EmaFilter(emaFx.alpha);
  (emaFx.frames as number[][][]).forEach((frame, i) => {
    const out = f.apply(frame);
    const exp = (emaFx.expected as number[][][])[i];
    out.forEach((lm, j) => lm.forEach((v, k) => expect(v).toBeCloseTo(exp[j][k], 6)));
  });
});

test('EmaFilter.reset restarts from next frame', () => {
  const f = new EmaFilter(0.5);
  f.apply([[0, 0, 0]]);
  f.reset();
  expect(f.apply([[1, 1, 1]])).toEqual([[1, 1, 1]]);
});

test('MajorityVote matches Python Counter behavior', () => {
  const mv = new MajorityVote(mvFx.maxlen);
  (mvFx.sequence as string[]).forEach((label, i) => {
    expect(mv.add(label)).toBe((mvFx.expected as string[])[i]);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd mobile && npm test -- smoothing` — Expected: FAIL (modules not found).

- [ ] **Step 3: Implement**

`mobile/src/core/ema.ts`:

```ts
export class EmaFilter {
  private state: number[][] | null = null;
  constructor(private readonly alpha: number = 0.5) {}

  apply(landmarks: number[][]): number[][] {
    if (this.state === null) {
      this.state = landmarks.map((lm) => [...lm]);
    } else {
      for (let i = 0; i < landmarks.length; i++) {
        for (let k = 0; k < landmarks[i].length; k++) {
          this.state[i][k] = this.alpha * landmarks[i][k] + (1 - this.alpha) * this.state[i][k];
        }
      }
    }
    return this.state.map((lm) => [...lm]);
  }

  reset(): void {
    this.state = null;
  }
}
```

`mobile/src/core/majorityVote.ts`:

```ts
export class MajorityVote {
  private history: string[] = [];
  constructor(private readonly maxlen: number = 10) {}

  add(label: string): string {
    this.history.push(label);
    if (this.history.length > this.maxlen) this.history.shift();
    const counts = new Map<string, number>();
    for (const l of this.history) counts.set(l, (counts.get(l) ?? 0) + 1);
    let best = this.history[0], bestCount = 0;
    for (const [l, c] of counts) {           // Map preserves first-seen order,
      if (c > bestCount) { best = l; bestCount = c; }  // matching Counter.most_common ties
    }
    return best;
  }

  reset(): void {
    this.history = [];
  }
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd mobile && npm test -- smoothing` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mobile/src/core/ema.ts mobile/src/core/majorityVote.ts mobile/src/core/__tests__/smoothing.test.ts
git commit -m "feat(core): port EMA smoothing and majority-vote with parity tests"
```

---

### Task 5: Core — exercise interface, BicepCurl, Squat

**Files:**
- Create: `mobile/src/core/exercises/types.ts`, `bicepCurl.ts`, `squat.ts`
- Test: `mobile/src/core/__tests__/exercises1.test.ts`

**Interfaces:**
- Produces:
  - `type Stage = 'up' | 'down' | 'bent' | 'straight'`
  - `type ExerciseAngles = Partial<JointAngles> & { active_arm?: 'left' | 'right' }`
  - `interface Exercise { repCount: number; stage: Stage; update(angles: ExerciseAngles, smoothedClass: string): [number, Stage] }`
  - `class BicepCurl implements Exercise`, `class Squat implements Exercise` — thresholds verbatim from `exercises.py` (curl: down >120/120, count <90/90; squat: up >140/140, count <115/115).

- [ ] **Step 1: Failing test**

`mobile/src/core/__tests__/exercises1.test.ts`:

```ts
import { BicepCurl } from '../exercises/bicepCurl';
import { Squat } from '../exercises/squat';
import fx from '../__fixtures__/exercises.json';
import type { Exercise, ExerciseAngles } from '../exercises/types';

type Step = { angles: ExerciseAngles; label: string; reps: number; stage: string };

function replay(machine: Exercise, steps: Step[]) {
  for (const s of steps) {
    const [reps, stage] = machine.update(s.angles, s.label);
    expect(reps).toBe(s.reps);
    expect(stage).toBe(s.stage);
  }
}

test('BicepCurl matches Python reference', () => {
  replay(new BicepCurl(), (fx as Record<string, Step[]>)['Bicep Curl']);
});

test('Squat matches Python reference', () => {
  replay(new Squat(), (fx as Record<string, Step[]>)['Squat']);
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd mobile && npm test -- exercises1` — Expected: FAIL.

- [ ] **Step 3: Implement**

`mobile/src/core/exercises/types.ts`:

```ts
import type { JointAngles } from '../angles';

export type Stage = 'up' | 'down' | 'bent' | 'straight';
export type ExerciseAngles = Partial<JointAngles> & { active_arm?: 'left' | 'right' };

export interface Exercise {
  repCount: number;
  stage: Stage;
  update(angles: ExerciseAngles, smoothedClass: string): [number, Stage];
}
```

`mobile/src/core/exercises/bicepCurl.ts`:

```ts
import type { Exercise, ExerciseAngles, Stage } from './types';

export class BicepCurl implements Exercise {
  repCount = 0;
  stage: Stage = 'down';
  private formWasBad = false;

  update(angles: ExerciseAngles, smoothedClass: string): [number, Stage] {
    const le = angles.l_elbow ?? 180;
    const re = angles.r_elbow ?? 180;
    if (smoothedClass.includes('Bad')) this.formWasBad = true;
    if (le > 120 && re > 120) {
      this.stage = 'down';
      this.formWasBad = false;
    }
    if (le < 90 && re < 90 && this.stage === 'down') {
      this.stage = 'up';
      if (!this.formWasBad && smoothedClass.includes('Good')) this.repCount++;
    }
    return [this.repCount, this.stage];
  }
}
```

`mobile/src/core/exercises/squat.ts`:

```ts
import type { Exercise, ExerciseAngles, Stage } from './types';

export class Squat implements Exercise {
  repCount = 0;
  stage: Stage = 'up';
  private formWasBad = false;

  update(angles: ExerciseAngles, smoothedClass: string): [number, Stage] {
    const lk = angles.l_knee ?? 180;
    const rk = angles.r_knee ?? 180;
    if (smoothedClass.includes('Bad')) this.formWasBad = true;
    if (lk > 140 && rk > 140) {
      this.stage = 'up';
      this.formWasBad = false;
    }
    if (lk < 115 && rk < 115 && this.stage === 'up') {
      this.stage = 'down';
      if (!this.formWasBad && smoothedClass.includes('Good')) this.repCount++;
    }
    return [this.repCount, this.stage];
  }
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd mobile && npm test -- exercises1` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mobile/src/core/exercises
git add mobile/src/core/__tests__/exercises1.test.ts
git commit -m "feat(core): port BicepCurl and Squat state machines"
```

---

### Task 6: Core — LateralRaise, ShoulderPress, TricepFinisher + registry

**Files:**
- Create: `mobile/src/core/exercises/lateralRaise.ts`, `shoulderPress.ts`, `tricepFinisher.ts`, `index.ts`
- Test: `mobile/src/core/__tests__/exercises2.test.ts`

**Interfaces:**
- Consumes: `Exercise`, `ExerciseAngles`, `Stage` from Task 5.
- Produces:
  - Three more `Exercise` classes (thresholds verbatim from `exercises.py`).
  - `type ExerciseName = 'Bicep Curl' | 'Squat' | 'Lateral Raise' | 'Shoulder Press' | 'Tricep Finisher'`
  - `EXERCISES: Record<ExerciseName, () => Exercise>` factory map (mirrors `EXERCISES_MAP` in `app.py`).

- [ ] **Step 1: Failing test**

`mobile/src/core/__tests__/exercises2.test.ts`:

```ts
import { EXERCISES, ExerciseName } from '../exercises';
import fx from '../__fixtures__/exercises.json';
import type { ExerciseAngles } from '../exercises/types';

type Step = { angles: ExerciseAngles; label: string; reps: number; stage: string };
const names: ExerciseName[] = ['Lateral Raise', 'Shoulder Press', 'Tricep Finisher'];

test.each(names)('%s matches Python reference', (name) => {
  const machine = EXERCISES[name]();
  for (const s of (fx as Record<string, Step[]>)[name]) {
    const [reps, stage] = machine.update(s.angles, s.label);
    expect(reps).toBe(s.reps);
    expect(stage).toBe(s.stage);
  }
});

test('registry has all five exercises', () => {
  expect(Object.keys(EXERCISES).sort()).toEqual(
    ['Bicep Curl', 'Lateral Raise', 'Shoulder Press', 'Squat', 'Tricep Finisher']);
});
```

- [ ] **Step 2: Run to verify failure** — `cd mobile && npm test -- exercises2` → FAIL.

- [ ] **Step 3: Implement**

`mobile/src/core/exercises/lateralRaise.ts`:

```ts
import type { Exercise, ExerciseAngles, Stage } from './types';

export class LateralRaise implements Exercise {
  repCount = 0;
  stage: Stage = 'down';
  private formWasBad = false;

  update(angles: ExerciseAngles, smoothedClass: string): [number, Stage] {
    const ls = angles.l_shoulder ?? 0;
    const rs = angles.r_shoulder ?? 0;
    if (smoothedClass.includes('Bad')) this.formWasBad = true;
    if (ls > 65 && rs > 65) this.stage = 'up';
    if (ls < 45 && rs < 45) {
      if (this.stage === 'up') {
        this.stage = 'down';
        if (!this.formWasBad && smoothedClass.includes('Good')) this.repCount++;
      } else {
        this.stage = 'down';
        this.formWasBad = false;
      }
    }
    return [this.repCount, this.stage];
  }
}
```

`mobile/src/core/exercises/shoulderPress.ts`:

```ts
import type { Exercise, ExerciseAngles, Stage } from './types';

export class ShoulderPress implements Exercise {
  repCount = 0;
  stage: Stage = 'down';
  private formWasBad = false;

  update(angles: ExerciseAngles, smoothedClass: string): [number, Stage] {
    const le = angles.l_elbow ?? 180;
    const re = angles.r_elbow ?? 180;
    if (smoothedClass.includes('Bad')) this.formWasBad = true;
    if (le < 100 && re < 100) {
      this.stage = 'down';
      this.formWasBad = false;
    }
    if (le > 150 && re > 150 && this.stage === 'down') {
      this.stage = 'up';
      if (!this.formWasBad && smoothedClass.includes('Good')) this.repCount++;
    }
    return [this.repCount, this.stage];
  }
}
```

`mobile/src/core/exercises/tricepFinisher.ts`:

```ts
import type { Exercise, ExerciseAngles, Stage } from './types';

export class TricepFinisher implements Exercise {
  repCount = 0;
  stage: Stage = 'bent';
  private formWasBad = false;

  update(angles: ExerciseAngles, smoothedClass: string): [number, Stage] {
    const arm = angles.active_arm ?? 'left';
    const el = (arm === 'left' ? angles.l_elbow : angles.r_elbow) ?? 180;
    if (smoothedClass.includes('Bad')) this.formWasBad = true;
    if (el < 70) {
      this.stage = 'bent';
      this.formWasBad = false;
    }
    if (el > 150 && this.stage === 'bent') {
      this.stage = 'straight';
      if (!this.formWasBad && smoothedClass.includes('Good')) this.repCount++;
    }
    return [this.repCount, this.stage];
  }
}
```

`mobile/src/core/exercises/index.ts`:

```ts
import type { Exercise } from './types';
import { BicepCurl } from './bicepCurl';
import { Squat } from './squat';
import { LateralRaise } from './lateralRaise';
import { ShoulderPress } from './shoulderPress';
import { TricepFinisher } from './tricepFinisher';

export type ExerciseName =
  | 'Bicep Curl' | 'Squat' | 'Lateral Raise' | 'Shoulder Press' | 'Tricep Finisher';

export const EXERCISES: Record<ExerciseName, () => Exercise> = {
  'Bicep Curl': () => new BicepCurl(),
  'Squat': () => new Squat(),
  'Lateral Raise': () => new LateralRaise(),
  'Shoulder Press': () => new ShoulderPress(),
  'Tricep Finisher': () => new TricepFinisher(),
};

export * from './types';
```

- [ ] **Step 4: Run all core tests** — `cd mobile && npm test` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add mobile/src/core/exercises mobile/src/core/__tests__/exercises2.test.ts
git commit -m "feat(core): port remaining state machines and exercise registry"
```

---

### Task 7: Core — classifier masking, softmax, labels

**Files:**
- Create: `mobile/src/core/classifier.ts`
- Test: `mobile/src/core/__tests__/classifier.test.ts`

**Interfaces:**
- Consumes: `ExerciseName` from Task 6; `JointAngles` from Task 3.
- Produces:
  - `LABELS: string[]` (index-ordered class labels)
  - `anglesToInput(a: JointAngles, exercise: ExerciseName): Float32Array` — length 8, order `[l_elbow, r_elbow, l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee]`, with the irrelevant group forced to 180 exactly as `app.py` does (upper-body exercises → hips/knees = 180; Squat → elbows/shoulders = 180).
  - `classify(logits: Float32Array | number[], exercise: ExerciseName): { label: string; confidence: number }` — mask to the exercise's class range, softmax, argmax; confidence in percent.

- [ ] **Step 1: Failing test**

`mobile/src/core/__tests__/classifier.test.ts`:

```ts
import { classify, anglesToInput, LABELS } from '../classifier';
import fx from '../__fixtures__/classifier.json';
import type { ExerciseName } from '../exercises';

type Row = { angles: number[]; logits: number[];
  verdicts: Record<string, { label: string; confidence: number }> };

test('classify matches Python masked softmax for every exercise', () => {
  for (const row of fx as Row[]) {
    for (const [ex, expected] of Object.entries(row.verdicts)) {
      const got = classify(row.logits, ex as ExerciseName);
      expect(got.label).toBe(expected.label);
      expect(got.confidence).toBeCloseTo(expected.confidence, 3);
    }
  }
});

test('anglesToInput zeroes the irrelevant group to 180', () => {
  const a = { l_elbow: 10, r_elbow: 20, l_shoulder: 30, r_shoulder: 40,
              l_hip: 50, r_hip: 60, l_knee: 70, r_knee: 80 };
  expect(Array.from(anglesToInput(a, 'Bicep Curl'))).toEqual([10, 20, 30, 40, 180, 180, 180, 180]);
  expect(Array.from(anglesToInput(a, 'Squat'))).toEqual([180, 180, 180, 180, 50, 60, 70, 80]);
});

test('LABELS ordering matches the model head', () => {
  expect(LABELS[1]).toBe('Good Curl');
  expect(LABELS[8]).toBe('Bad Tricep');
});
```

- [ ] **Step 2: Run to verify failure** — `cd mobile && npm test -- classifier` → FAIL.

- [ ] **Step 3: Implement**

`mobile/src/core/classifier.ts`:

```ts
import type { JointAngles } from './angles';
import type { ExerciseName } from './exercises';

export const LABELS = [
  'Bad Curl', 'Good Curl', 'Bad Squat', 'Good Squat', 'Bad Raise', 'Good Raise',
  'Bad Shoulder', 'Good Shoulder', 'Bad Tricep', 'Good Tricep',
];

const MASK_RANGES: Record<ExerciseName, [number, number]> = {
  'Bicep Curl': [0, 2],
  'Squat': [2, 4],
  'Lateral Raise': [4, 6],
  'Shoulder Press': [6, 8],
  'Tricep Finisher': [8, 10],
};

const UPPER_BODY: ExerciseName[] = ['Bicep Curl', 'Lateral Raise', 'Shoulder Press', 'Tricep Finisher'];

export function anglesToInput(a: JointAngles, exercise: ExerciseName): Float32Array {
  const v = { ...a };
  if (UPPER_BODY.includes(exercise)) {
    v.l_hip = v.r_hip = v.l_knee = v.r_knee = 180;
  } else {
    v.l_elbow = v.r_elbow = v.l_shoulder = v.r_shoulder = 180;
  }
  return Float32Array.from([
    v.l_elbow, v.r_elbow, v.l_shoulder, v.r_shoulder, v.l_hip, v.r_hip, v.l_knee, v.r_knee,
  ]);
}

export function classify(
  logits: Float32Array | number[],
  exercise: ExerciseName,
): { label: string; confidence: number } {
  const [lo, hi] = MASK_RANGES[exercise];
  const slice = Array.from(logits).slice(lo, hi);
  const m = Math.max(...slice);
  const exps = slice.map((x) => Math.exp(x - m));
  const sum = exps.reduce((s, x) => s + x, 0);
  let bestI = 0;
  for (let i = 1; i < exps.length; i++) if (exps[i] > exps[bestI]) bestI = i;
  return { label: LABELS[lo + bestI], confidence: (exps[bestI] / sum) * 100 };
}
```

- [ ] **Step 4: Run to verify pass** — `cd mobile && npm test -- classifier` → PASS.

- [ ] **Step 5: Commit**

```bash
git add mobile/src/core/classifier.ts mobile/src/core/__tests__/classifier.test.ts
git commit -m "feat(core): classifier input mapping and masked softmax"
```

---

### Task 8: Export the model to ONNX with Python parity check

**Files:**
- Create: `tools/export_onnx.py`
- Create (generated): `mobile/assets/models/gym_model_fullbody.onnx`

**Interfaces:**
- Consumes: `gym_model_fullbody.pt`, `GymModel` architecture (8→64→32→10).
- Produces: ONNX model, input name `angles` shape `[1, 8]` float32, output name `logits` shape `[1, 10]`.

- [ ] **Step 1: Write the export + verification script**

`tools/export_onnx.py`:

```python
"""Export gym_model_fullbody.pt to ONNX and verify output parity."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mobile" / "assets" / "models" / "gym_model_fullbody.onnx"
OUT.parent.mkdir(parents=True, exist_ok=True)

class GymModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    def forward(self, x):
        return self.network(x)

model = GymModel()
model.load_state_dict(torch.load(ROOT / "gym_model_fullbody.pt", weights_only=True))
model.eval()

torch.onnx.export(
    model, torch.randn(1, 8), str(OUT),
    input_names=["angles"], output_names=["logits"], opset_version=17)

# Parity check
import onnxruntime as ort  # pip install onnxruntime
sess = ort.InferenceSession(str(OUT))
rng = np.random.default_rng(42)
x = rng.uniform(20, 180, size=(50, 8)).astype(np.float32)
with torch.no_grad():
    torch_out = model(torch.from_numpy(x)).numpy()
onnx_out = np.concatenate([sess.run(None, {"angles": x[i:i+1]})[0] for i in range(len(x))])
max_diff = float(np.abs(torch_out - onnx_out).max())
assert max_diff < 1e-4, f"ONNX diverges from PyTorch: max diff {max_diff}"
print(f"OK: exported {OUT}, max diff vs PyTorch = {max_diff:.2e}")
```

- [ ] **Step 2: Run it**

Run: `pip install onnx onnxruntime && python tools/export_onnx.py`
Expected: `OK: exported ... max diff vs PyTorch = ...` (< 1e-4).

- [ ] **Step 3: Commit**

```bash
git add tools/export_onnx.py mobile/assets/models/gym_model_fullbody.onnx
git commit -m "feat(tools): export classifier to ONNX with parity check"
```

---

### Task 9: ONNX inference in TypeScript with golden-file tests

**Files:**
- Create: `mobile/src/ml/onnxClassifier.ts`
- Test: `mobile/src/ml/__tests__/onnxClassifier.test.ts`

**Interfaces:**
- Consumes: ONNX model from Task 8; `classify`, `anglesToInput` from Task 7.
- Produces: `createOnnxClassifier(modelPath: string, ort: OrtLike): Promise<OnnxClassifier>` where `OnnxClassifier = { run(input: Float32Array): Promise<Float32Array> }`. `OrtLike` is a minimal structural type `{ InferenceSession: { create(path: string): Promise<...> }, Tensor: new (...) }` satisfied by **both** `onnxruntime-node` (Jest/CI) and `onnxruntime-react-native` (app) — the app injects whichever runtime it has.

- [ ] **Step 1: Install runtimes**

```bash
cd mobile
npm install onnxruntime-react-native
npm install --save-dev onnxruntime-node
```

- [ ] **Step 2: Failing golden test**

`mobile/src/ml/__tests__/onnxClassifier.test.ts`:

```ts
import * as ortNode from 'onnxruntime-node';
import * as path from 'path';
import { createOnnxClassifier } from '../onnxClassifier';
import { classify } from '../../core/classifier';
import fx from '../../core/__fixtures__/classifier.json';
import type { ExerciseName } from '../../core/exercises';

type Row = { angles: number[]; logits: number[];
  verdicts: Record<string, { label: string; confidence: number }> };

const MODEL = path.resolve(__dirname, '../../../assets/models/gym_model_fullbody.onnx');

test('ONNX logits and end-to-end verdicts match the PyTorch reference', async () => {
  const clf = await createOnnxClassifier(MODEL, ortNode);
  for (const row of (fx as Row[]).slice(0, 25)) {
    const logits = await clf.run(Float32Array.from(row.angles));
    row.logits.forEach((expected, i) => expect(logits[i]).toBeCloseTo(expected, 3));
    for (const [ex, v] of Object.entries(row.verdicts)) {
      expect(classify(logits, ex as ExerciseName).label).toBe(v.label);
    }
  }
});
```

Run: `cd mobile && npm test -- onnxClassifier` — Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

`mobile/src/ml/onnxClassifier.ts`:

```ts
interface OrtTensor { data: Float32Array }
interface OrtSession { run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensor>> }
export interface OrtLike {
  InferenceSession: { create(path: string): Promise<OrtSession> };
  Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => unknown;
}

export interface OnnxClassifier {
  run(input: Float32Array): Promise<Float32Array>;
}

export async function createOnnxClassifier(modelPath: string, ort: OrtLike): Promise<OnnxClassifier> {
  const session = await ort.InferenceSession.create(modelPath);
  return {
    async run(input: Float32Array): Promise<Float32Array> {
      const feeds = { angles: new ort.Tensor('float32', input, [1, 8]) };
      const out = await session.run(feeds);
      return out['logits'].data;
    },
  };
}
```

- [ ] **Step 4: Run to verify pass** — `cd mobile && npm test -- onnxClassifier` → PASS.

- [ ] **Step 5: Commit**

```bash
git add mobile/src/ml mobile/package.json mobile/package-lock.json
git commit -m "feat(ml): ONNX classifier wrapper with golden-file parity tests"
```

---

### Task 10: Camera preview with VisionCamera

**Files:**
- Create: `mobile/src/screens/CoachScreen.tsx`
- Modify: `mobile/App.tsx`, `mobile/app.json`

**Interfaces:**
- Produces: `CoachScreen` rendering a full-screen front-camera preview at 480×360 target with permission handling; `PermissionGate` pattern reused by later tasks.

- [ ] **Step 1: Install and configure**

```bash
cd mobile
npm install react-native-vision-camera react-native-reanimated react-native-worklets-core
```

In `mobile/app.json` add the plugin + permission strings:

```json
"plugins": [
  ["react-native-vision-camera", {
    "cameraPermissionText": "GymForm AI uses the camera to track your exercise form. Video never leaves your device."
  }]
]
```

- [ ] **Step 2: Implement the screen**

`mobile/src/screens/CoachScreen.tsx`:

```tsx
import React, { useEffect } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { Camera, useCameraDevice, useCameraFormat, useCameraPermission } from 'react-native-vision-camera';

export function CoachScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('front');
  const format = useCameraFormat(device, [
    { videoResolution: { width: 480, height: 360 } },
    { fps: 30 },
  ]);

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.msg}>Camera access is needed to coach your form.</Text>
        <Text style={styles.msg}>Enable it in Settings, then reopen the app.</Text>
      </View>
    );
  }
  if (device == null) {
    return <View style={styles.center}><Text style={styles.msg}>No camera found.</Text></View>;
  }
  return (
    <Camera style={StyleSheet.absoluteFill} device={device} format={format} isActive={true} />
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#0a0a0a' },
  msg: { color: '#fff', margin: 4 },
});
```

Point `App.tsx` at it:

```tsx
import { CoachScreen } from './src/screens/CoachScreen';
export default function App() {
  return <CoachScreen />;
}
```

- [ ] **Step 3: Build dev client and verify on device**

```bash
cd mobile && npx expo prebuild && npx expo run:ios   # and/or run:android
```

Expected: app asks for camera permission, then shows a live front-camera preview. (Simulators have no camera — use a physical device from here on.)

- [ ] **Step 4: Commit**

```bash
git add mobile
git commit -m "feat(app): camera preview screen with VisionCamera"
```

---

### Task 11: Android pose landmarker frame-processor plugin

**Files:**
- Create: `mobile/android/app/src/main/java/.../posedetector/PoseFrameProcessorPlugin.kt`, `PoseFrameProcessorPluginPackage.kt`
- Modify: `mobile/android/app/build.gradle` (MediaPipe dependency), `MainApplication` package registration
- Create: `mobile/android/app/src/main/assets/pose_landmarker_lite.task` (copy of root `pose_landmarker_lite.task`)

**Interfaces:**
- Produces: a VisionCamera frame-processor plugin registered as `"detectPose"`. Return value per frame: `null` if no pose, else `Map` `{ landmarks: [{x, y, z, visibility} × 33] }` in normalized image coordinates. This exact shape is consumed by Task 13.

- [ ] **Step 1: Add the MediaPipe dependency**

In `mobile/android/app/build.gradle` dependencies:

```groovy
implementation 'com.google.mediapipe:tasks-vision:0.10.14'
```

Copy the model: `cp pose_landmarker_lite.task mobile/android/app/src/main/assets/`.

- [ ] **Step 2: Implement the plugin**

`PoseFrameProcessorPlugin.kt` (package path per the generated project's applicationId):

```kotlin
package com.gymform.mobile.posedetector

import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.mrousavy.camera.frameprocessors.Frame
import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.VisionCameraProxy

class PoseFrameProcessorPlugin(proxy: VisionCameraProxy, options: Map<String, Any>?) :
  FrameProcessorPlugin() {

  private val landmarker: PoseLandmarker by lazy {
    val base = BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").build()
    val opts = PoseLandmarker.PoseLandmarkerOptions.builder()
      .setBaseOptions(base)
      .setRunningMode(RunningMode.IMAGE)
      .setNumPoses(1)
      .build()
    PoseLandmarker.createFromOptions(proxy.context, opts)
  }

  override fun callback(frame: Frame, arguments: Map<String, Any>?): Any? {
    val bitmap = frame.imageProxy.toBitmap()  // includes rotation handling via imageInfo
    val mpImage = BitmapImageBuilder(bitmap).build()
    val result = landmarker.detect(mpImage)
    if (result.landmarks().isEmpty()) return null
    val lms = result.landmarks()[0].map { lm ->
      mapOf(
        "x" to lm.x().toDouble(),
        "y" to lm.y().toDouble(),
        "z" to lm.z().toDouble(),
        "visibility" to (lm.visibility().orElse(0.0f)).toDouble(),
      )
    }
    return mapOf("landmarks" to lms)
  }
}
```

`PoseFrameProcessorPluginPackage.kt` — register with VisionCamera:

```kotlin
package com.gymform.mobile.posedetector

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager
import com.mrousavy.camera.frameprocessors.FrameProcessorPluginRegistry

class PoseFrameProcessorPluginPackage : ReactPackage {
  companion object {
    init {
      FrameProcessorPluginRegistry.addFrameProcessorPlugin("detectPose") { proxy, options ->
        PoseFrameProcessorPlugin(proxy, options)
      }
    }
  }
  override fun createNativeModules(ctx: ReactApplicationContext): List<NativeModule> = emptyList()
  override fun createViewManagers(ctx: ReactApplicationContext): List<ViewManager<*, *>> = emptyList()
}
```

Add `PoseFrameProcessorPluginPackage()` to the packages list in `MainApplication`.

Note: this task's files live under `android/`, which `expo prebuild` can regenerate. Also add an Expo config-plugin note in `mobile/README.md`: "after `expo prebuild --clean`, re-apply the posedetector package registration" (or migrate to a local Expo module later — out of scope for Phase 1).

- [ ] **Step 3: Verify on device**

Temporarily add to `CoachScreen` a frame processor logging detection:

```tsx
import { useFrameProcessor, VisionCameraProxy } from 'react-native-vision-camera';
const plugin = VisionCameraProxy.initFrameProcessorPlugin('detectPose', {});
const frameProcessor = useFrameProcessor((frame) => {
  'worklet';
  const result = plugin?.call(frame);
  if (result != null) console.log('pose detected: 33 landmarks');
}, []);
// pass frameProcessor={frameProcessor} to <Camera />
```

Run: `npx expo run:android`, stand in frame. Expected: "pose detected" logs stream in Metro. Remove the temporary logging after verification (keep the plugin init — Task 13 builds on it).

- [ ] **Step 4: Commit**

```bash
git add mobile/android mobile/src/screens/CoachScreen.tsx mobile/README.md
git commit -m "feat(android): MediaPipe pose landmarker frame-processor plugin"
```

---

### Task 12: iOS pose landmarker frame-processor plugin

**Files:**
- Create: `mobile/ios/PoseFrameProcessorPlugin.swift`, `mobile/ios/PoseFrameProcessorPlugin.m`
- Modify: `mobile/ios/Podfile` (`pod 'MediaPipeTasksVision'`), Xcode project (bundle `pose_landmarker_lite.task` as a resource)

**Interfaces:**
- Produces: the same `"detectPose"` plugin contract as Task 11 — `null` or `{ landmarks: [{x, y, z, visibility} × 33] }` — so Task 13's TS code is platform-agnostic.

- [ ] **Step 1: Add the pod and model resource**

In `mobile/ios/Podfile` (app target): `pod 'MediaPipeTasksVision'`, then `cd mobile/ios && pod install`. Add `pose_landmarker_lite.task` (copy from repo root) to the Xcode project as a bundle resource.

- [ ] **Step 2: Implement the plugin**

`mobile/ios/PoseFrameProcessorPlugin.swift`:

```swift
import Foundation
import MediaPipeTasksVision
import VisionCamera

@objc(PoseFrameProcessorPlugin)
public class PoseFrameProcessorPlugin: FrameProcessorPlugin {
  private static let landmarker: PoseLandmarker? = {
    guard let modelPath = Bundle.main.path(forResource: "pose_landmarker_lite", ofType: "task")
    else { return nil }
    let options = PoseLandmarkerOptions()
    options.baseOptions.modelAssetPath = modelPath
    options.runningMode = .image
    options.numPoses = 1
    return try? PoseLandmarker(options: options)
  }()

  public override func callback(_ frame: Frame, withArguments arguments: [AnyHashable: Any]?) -> Any? {
    guard let landmarker = Self.landmarker,
          let image = try? MPImage(sampleBuffer: frame.buffer,
                                   orientation: frame.orientation),
          let result = try? landmarker.detect(image: image),
          let pose = result.landmarks.first
    else { return nil }

    let lms: [[String: Double]] = pose.map { lm in
      ["x": Double(lm.x), "y": Double(lm.y), "z": Double(lm.z),
       "visibility": Double(truncating: lm.visibility ?? 0)]
    }
    return ["landmarks": lms]
  }
}
```

`mobile/ios/PoseFrameProcessorPlugin.m`:

```objc
#import <VisionCamera/FrameProcessorPlugin.h>
#import <VisionCamera/FrameProcessorPluginRegistry.h>
#import "GymFormMobile-Swift.h"

VISION_EXPORT_SWIFT_FRAME_PROCESSOR(PoseFrameProcessorPlugin, detectPose)
```

(Adjust the `-Swift.h` header name to the generated project's module name.)

- [ ] **Step 3: Verify on device**

Run: `npx expo run:ios --device` with the same temporary logging as Task 11 Step 3. Expected: "pose detected" logs.

- [ ] **Step 4: Commit**

```bash
git add mobile/ios
git commit -m "feat(ios): MediaPipe pose landmarker frame-processor plugin"
```

---

### Task 13: Pipeline wiring — landmarks to coach state

**Files:**
- Create: `mobile/src/coach/pipeline.ts`, `mobile/src/coach/useCoach.ts`
- Test: `mobile/src/coach/__tests__/pipeline.test.ts`
- Modify: `mobile/src/screens/CoachScreen.tsx`

**Interfaces:**
- Consumes: Tasks 3–7 core modules; Task 9 `OnnxClassifier`; Tasks 11–12 plugin result shape.
- Produces:
  - `type PoseResult = { landmarks: { x: number; y: number; z: number; visibility: number }[] }`
  - `type CoachState = { reps: number; stage: string; formLabel: string; confidence: number; smoothedLandmarks: number[][] | null; badJoints: number[] }`
  - `class CoachPipeline { constructor(exercise: ExerciseName, classifier: OnnxClassifier); process(pose: PoseResult | null): Promise<CoachState>; setExercise(e: ExerciseName): void }` — `setExercise` resets EMA, vote history, and state machine (mirrors the setter in `app.py`).
  - `useCoach(exercise: ExerciseName)` React hook exposing `{ state, onPose }` where `onPose` is called from the frame-processor bridge (via `runOnJS`), with backpressure: if a classify call is in flight, the frame is dropped.

- [ ] **Step 1: Failing test (pipeline only — hook is exercised on device)**

`mobile/src/coach/__tests__/pipeline.test.ts`:

```ts
import { CoachPipeline } from '../pipeline';
import type { OnnxClassifier } from '../../ml/onnxClassifier';

// Fake classifier: high "Good Curl" logit regardless of input.
const goodCurlClassifier: OnnxClassifier = {
  run: async () => Float32Array.from([0, 10, 0, 0, 0, 0, 0, 0, 0, 0]),
};

function poseWithElbowAngle(deg: number) {
  // Build 33 landmarks where l/r elbow angles ≈ deg by placing wrist accordingly.
  const lms = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 1 }));
  const rad = (deg * Math.PI) / 180;
  // shoulder (11/12) above elbow (13/14); wrist (15/16) rotated by `deg` around elbow
  lms[11] = { x: 0.5, y: 0.3, z: 0, visibility: 1 };
  lms[12] = { x: 0.7, y: 0.3, z: 0, visibility: 1 };
  lms[13] = { x: 0.5, y: 0.5, z: 0, visibility: 1 };
  lms[14] = { x: 0.7, y: 0.5, z: 0, visibility: 1 };
  lms[15] = { x: 0.5 + 0.2 * Math.sin(rad), y: 0.5 + 0.2 * Math.cos(rad), z: 0, visibility: 1 };
  lms[16] = { x: 0.7 + 0.2 * Math.sin(rad), y: 0.5 + 0.2 * Math.cos(rad), z: 0, visibility: 1 };
  return { landmarks: lms };
}

test('a full good curl counts one rep end-to-end', async () => {
  const p = new CoachPipeline('Bicep Curl', goodCurlClassifier);
  // Feed enough frames for EMA + majority vote to settle, down then up.
  for (let i = 0; i < 15; i++) await p.process(poseWithElbowAngle(170)); // arms extended
  let s;
  for (let i = 0; i < 15; i++) s = await p.process(poseWithElbowAngle(30)); // curled
  expect(s!.formLabel).toBe('Good Curl');
  expect(s!.reps).toBe(1);
  expect(s!.stage).toBe('up');
});

test('no pose yields null landmarks and preserved state', async () => {
  const p = new CoachPipeline('Bicep Curl', goodCurlClassifier);
  const s = await p.process(null);
  expect(s.smoothedLandmarks).toBeNull();
  expect(s.reps).toBe(0);
});

test('setExercise resets counters', async () => {
  const p = new CoachPipeline('Bicep Curl', goodCurlClassifier);
  for (let i = 0; i < 15; i++) await p.process(poseWithElbowAngle(170));
  for (let i = 0; i < 15; i++) await p.process(poseWithElbowAngle(30));
  p.setExercise('Squat');
  const s = await p.process(poseWithElbowAngle(170));
  expect(s.reps).toBe(0);
});
```

Run: `cd mobile && npm test -- pipeline` — Expected: FAIL.

- [ ] **Step 2: Implement the pipeline**

`mobile/src/coach/pipeline.ts`:

```ts
import { computeJointAngles } from '../core/angles';
import { EmaFilter } from '../core/ema';
import { MajorityVote } from '../core/majorityVote';
import { anglesToInput, classify } from '../core/classifier';
import { EXERCISES, type Exercise, type ExerciseName, type ExerciseAngles } from '../core/exercises';
import type { OnnxClassifier } from '../ml/onnxClassifier';

export interface PoseResult {
  landmarks: { x: number; y: number; z: number; visibility: number }[];
}

export interface CoachState {
  reps: number;
  stage: string;
  formLabel: string;
  confidence: number;
  smoothedLandmarks: number[][] | null;
  badJoints: number[];
}

const CONFIDENCE_FLOOR = 60;

export class CoachPipeline {
  private ema = new EmaFilter(0.5);
  private vote = new MajorityVote(10);
  private machine: Exercise;
  private state: CoachState = {
    reps: 0, stage: 'down', formLabel: 'Waiting...', confidence: 0,
    smoothedLandmarks: null, badJoints: [],
  };

  constructor(private exercise: ExerciseName, private classifier: OnnxClassifier) {
    this.machine = EXERCISES[exercise]();
  }

  setExercise(e: ExerciseName): void {
    this.exercise = e;
    this.machine = EXERCISES[e]();
    this.ema.reset();
    this.vote.reset();
    this.state = { reps: 0, stage: this.machine.stage, formLabel: 'Waiting...',
                   confidence: 0, smoothedLandmarks: null, badJoints: [] };
  }

  async process(pose: PoseResult | null): Promise<CoachState> {
    if (pose === null) {
      return { ...this.state, smoothedLandmarks: null };
    }
    const raw = pose.landmarks.map((lm) => [lm.x, lm.y, lm.z]);
    const smoothed = this.ema.apply(raw);
    const angles = computeJointAngles(smoothed);

    const exAngles: ExerciseAngles = {
      ...angles,
      active_arm:
        (pose.landmarks[13]?.visibility ?? 0) > (pose.landmarks[14]?.visibility ?? 0)
          ? 'left' : 'right',
    };

    const logits = await this.classifier.run(anglesToInput(angles, this.exercise));
    const { label, confidence } = classify(logits, this.exercise);
    let smoothedLabel = this.vote.add(label);
    if (confidence < CONFIDENCE_FLOOR) smoothedLabel = 'Tracking...';

    const [reps, stage] = this.machine.update(exAngles, smoothedLabel);
    const badJoints = smoothedLabel.includes('Bad')
      ? this.badJointsFor(exAngles) : [];

    this.state = { reps, stage, formLabel: smoothedLabel, confidence,
                   smoothedLandmarks: smoothed, badJoints };
    return this.state;
  }

  // Joint indices to highlight red, condensed from the per-exercise blocks in app.py.
  private badJointsFor(a: ExerciseAngles): number[] {
    switch (this.exercise) {
      case 'Bicep Curl':
        return (a.l_shoulder ?? 0) > 40 || (a.r_shoulder ?? 0) > 40
          ? [11, 12, 13, 14, 23, 24] : [11, 12, 13, 14, 15, 16];
      case 'Squat':
        return (a.l_hip ?? 180) < 70 || (a.r_hip ?? 180) < 70
          ? [11, 12, 23, 24] : [23, 24, 25, 26, 27, 28];
      case 'Lateral Raise':
        return (a.l_elbow ?? 180) < 140 || (a.r_elbow ?? 180) < 140
          ? [11, 12, 13, 14, 15, 16] : [11, 12, 13, 14];
      case 'Shoulder Press':
        return (a.l_hip ?? 180) < 160 || (a.r_hip ?? 180) < 160
          ? [11, 12, 23, 24] : [11, 12, 13, 14, 15, 16];
      case 'Tricep Finisher':
        return (a.l_shoulder ?? 0) > 45 && (a.r_shoulder ?? 0) > 45
          ? [11, 12, 13, 14] : [11, 12, 13, 14, 15, 16];
    }
  }
}
```

- [ ] **Step 3: Run to verify pass** — `cd mobile && npm test -- pipeline` → PASS.

- [ ] **Step 4: The hook + screen wiring**

`mobile/src/coach/useCoach.ts`:

```ts
import { useCallback, useEffect, useRef, useState } from 'react';
import * as ort from 'onnxruntime-react-native';
import { Asset } from 'expo-asset';
import { CoachPipeline, type CoachState, type PoseResult } from './pipeline';
import { createOnnxClassifier, type OrtLike } from '../ml/onnxClassifier';
import type { ExerciseName } from '../core/exercises';

const IDLE: CoachState = { reps: 0, stage: 'down', formLabel: 'Waiting...',
  confidence: 0, smoothedLandmarks: null, badJoints: [] };

export function useCoach(exercise: ExerciseName) {
  const pipelineRef = useRef<CoachPipeline | null>(null);
  const busyRef = useRef(false);
  const [state, setState] = useState<CoachState>(IDLE);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const asset = Asset.fromModule(require('../../assets/models/gym_model_fullbody.onnx'));
      await asset.downloadAsync();
      const clf = await createOnnxClassifier(asset.localUri!, ort as unknown as OrtLike);
      if (!cancelled) pipelineRef.current = new CoachPipeline(exercise, clf);
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    pipelineRef.current?.setExercise(exercise);
    setState(IDLE);
  }, [exercise]);

  const onPose = useCallback((pose: PoseResult | null) => {
    const pipeline = pipelineRef.current;
    if (pipeline == null || busyRef.current) return;  // backpressure: drop frame
    busyRef.current = true;
    pipeline.process(pose)
      .then(setState)
      .finally(() => { busyRef.current = false; });
  }, []);

  return { state, onPose };
}
```

In `CoachScreen.tsx`, wire the frame processor to the hook:

```tsx
import { Worklets } from 'react-native-worklets-core';
// inside component:
const { state, onPose } = useCoach(exercise);
const onPoseJS = Worklets.createRunOnJS(onPose);
const frameProcessor = useFrameProcessor((frame) => {
  'worklet';
  const result = plugin?.call(frame) as unknown as PoseResult | null;
  onPoseJS(result ?? null);
}, [onPoseJS]);
```

Add `metro.config.js` asset extension so the model bundles: `config.resolver.assetExts.push('onnx', 'task')`. Install `expo-asset` (`npx expo install expo-asset`).

- [ ] **Step 5: Verify on device**

Run the app; perform curls in frame. Expected: Metro logs / temporary on-screen text show reps incrementing and labels changing. 

- [ ] **Step 6: Commit**

```bash
git add mobile/src/coach mobile/src/screens/CoachScreen.tsx mobile/metro.config.js mobile/package.json mobile/package-lock.json
git commit -m "feat(coach): end-to-end pipeline from pose to reps with backpressure"
```

---

### Task 14: Skia skeleton overlay

**Files:**
- Create: `mobile/src/overlay/connections.ts`, `mobile/src/overlay/SkeletonOverlay.tsx`
- Test: `mobile/src/overlay/__tests__/connections.test.ts`
- Modify: `mobile/src/screens/CoachScreen.tsx`

**Interfaces:**
- Consumes: `CoachState` from Task 13.
- Produces:
  - `POSE_CONNECTIONS: [number, number][]` (verbatim from `app.py`)
  - `visibleConnections(exercise: ExerciseName): [number, number][]` and `visibleJoints(exercise: ExerciseName): number[]` — filtering rules from `app.py`: skip indices ≤10 (face); upper-body exercises skip ≥25; Squat skips 13–22.
  - `SkeletonOverlay({ state, exercise, width, height })` — Skia canvas: lines 3px in green (`#00FF00`) / cyan when `formLabel === 'Tracking...'`; connections touching `badJoints` drawn red (`#FF0000`) 6px; joints as 5px circles, white or red.

- [ ] **Step 1: Failing test (pure filtering logic)**

`mobile/src/overlay/__tests__/connections.test.ts`:

```ts
import { POSE_CONNECTIONS, visibleConnections, visibleJoints } from '../connections';

test('face connections are always excluded', () => {
  for (const [s, e] of visibleConnections('Squat')) {
    expect(s).toBeGreaterThan(10);
    expect(e).toBeGreaterThan(10);
  }
});

test('upper-body exercises exclude legs (>=25)', () => {
  for (const [s, e] of visibleConnections('Bicep Curl')) {
    expect(s).toBeLessThan(25);
    expect(e).toBeLessThan(25);
  }
  expect(visibleJoints('Bicep Curl').every((i) => i > 10 && i < 25)).toBe(true);
});

test('squat excludes arms/hands (13-22)', () => {
  for (const [s, e] of visibleConnections('Squat')) {
    expect(s < 13 || s > 22).toBe(true);
    expect(e < 13 || e > 22).toBe(true);
  }
});

test('POSE_CONNECTIONS matches the reference count', () => {
  expect(POSE_CONNECTIONS.length).toBe(35);
});
```

Run: `cd mobile && npm test -- connections` — Expected: FAIL.

- [ ] **Step 2: Implement `connections.ts`**

```ts
import type { ExerciseName } from '../core/exercises';

export const POSE_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
  [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], [28, 30],
  [29, 31], [30, 32], [27, 31], [28, 32],
];

const UPPER_BODY: ExerciseName[] = ['Bicep Curl', 'Lateral Raise', 'Shoulder Press', 'Tricep Finisher'];

function jointVisible(i: number, exercise: ExerciseName): boolean {
  if (i <= 10) return false;
  if (UPPER_BODY.includes(exercise) && i >= 25) return false;
  if (exercise === 'Squat' && i >= 13 && i <= 22) return false;
  return true;
}

export function visibleConnections(exercise: ExerciseName): [number, number][] {
  return POSE_CONNECTIONS.filter(([s, e]) => jointVisible(s, exercise) && jointVisible(e, exercise));
}

export function visibleJoints(exercise: ExerciseName): number[] {
  return Array.from({ length: 33 }, (_, i) => i).filter((i) => jointVisible(i, exercise));
}
```

- [ ] **Step 3: Run to verify pass** — `cd mobile && npm test -- connections` → PASS.

- [ ] **Step 4: Implement the overlay component**

```bash
cd mobile && npx expo install @shopify/react-native-skia
```

`mobile/src/overlay/SkeletonOverlay.tsx`:

```tsx
import React from 'react';
import { Canvas, Circle, Line, vec } from '@shopify/react-native-skia';
import { visibleConnections, visibleJoints } from './connections';
import type { CoachState } from '../coach/pipeline';
import type { ExerciseName } from '../core/exercises';

interface Props { state: CoachState; exercise: ExerciseName; width: number; height: number }

export function SkeletonOverlay({ state, exercise, width, height }: Props) {
  const lms = state.smoothedLandmarks;
  if (lms == null) return null;
  const tracking = state.formLabel === 'Tracking...';
  const baseColor = tracking ? '#00FFFF' : '#00FF00';
  const bad = new Set(state.badJoints);

  return (
    <Canvas style={{ position: 'absolute', width, height }} pointerEvents="none">
      {visibleConnections(exercise).map(([s, e]) => {
        const isBad = bad.has(s) && bad.has(e);
        return (
          <Line
            key={`${s}-${e}`}
            p1={vec(lms[s][0] * width, lms[s][1] * height)}
            p2={vec(lms[e][0] * width, lms[e][1] * height)}
            color={isBad ? '#FF0000' : baseColor}
            strokeWidth={isBad ? 6 : 3}
          />
        );
      })}
      {visibleJoints(exercise).map((i) => (
        <Circle
          key={i}
          cx={lms[i][0] * width}
          cy={lms[i][1] * height}
          r={5}
          color={bad.has(i) ? '#FF0000' : '#FFFFFF'}
        />
      ))}
    </Canvas>
  );
}
```

Mount it in `CoachScreen` over the camera (measure the camera view with `onLayout` for `width`/`height`). Front camera preview is mirrored — mirror x (`(1 - lms[i][0]) * width`) if the overlay appears flipped on device; verify visually.

- [ ] **Step 5: Verify on device** — skeleton tracks the body; goes red-jointed on deliberate bad form (swing shoulders during a curl).

- [ ] **Step 6: Commit**

```bash
git add mobile/src/overlay mobile/src/screens/CoachScreen.tsx mobile/package.json mobile/package-lock.json
git commit -m "feat(overlay): Skia skeleton with bad-joint highlighting"
```

---

### Task 15: Coach stats UI

**Files:**
- Create: `mobile/src/screens/components/StatCard.tsx`
- Modify: `mobile/src/screens/CoachScreen.tsx`

**Interfaces:**
- Consumes: `CoachState` from Task 13.
- Produces: `StatCard({ label, value, tone })` with `tone: 'good' | 'bad' | 'neutral' | 'default'`; CoachScreen renders three cards (Reps / Stage / Form + confidence) under the camera, updating at state-change rate (React state from `useCoach` — already decoupled from frame rate by backpressure).

- [ ] **Step 1: Implement `StatCard`**

```tsx
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

const toneColor = { good: '#39FF14', bad: '#FF4444', neutral: '#FFD700', default: '#00D2FF' } as const;

interface Props { label: string; value: string; tone?: keyof typeof toneColor }

export function StatCard({ label, value, tone = 'default' }: Props) {
  return (
    <View style={styles.card}>
      <Text style={styles.label}>{label.toUpperCase()}</Text>
      <Text style={[styles.value, { color: toneColor[tone] }]}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: { flex: 1, margin: 4, padding: 10, borderRadius: 10, alignItems: 'center',
          backgroundColor: 'rgba(255,255,255,0.06)', borderWidth: 1,
          borderColor: 'rgba(57,255,20,0.3)' },
  label: { color: 'rgba(255,255,255,0.45)', fontSize: 10, fontWeight: '700', letterSpacing: 1.5 },
  value: { fontSize: 22, fontWeight: '800', marginTop: 2 },
});
```

- [ ] **Step 2: Wire into CoachScreen**

Below the camera view:

```tsx
<View style={{ flexDirection: 'row', padding: 8, backgroundColor: '#0a0a0a' }}>
  <StatCard label="Reps" value={String(state.reps)} />
  <StatCard label="Stage" value={state.stage.toUpperCase()} />
  <StatCard
    label="Form"
    value={`${state.formLabel} (${state.confidence.toFixed(0)}%)`}
    tone={state.formLabel.includes('Good') ? 'good'
        : state.formLabel.includes('Bad') ? 'bad' : 'neutral'}
  />
</View>
```

- [ ] **Step 3: Verify on device** — cards update live during a set; form card colors match verdicts.

- [ ] **Step 4: Commit**

```bash
git add mobile/src/screens
git commit -m "feat(ui): live stats cards on coach screen"
```

---

### Task 16: Guided session flow

**Files:**
- Create: `mobile/src/session/sessionMachine.ts`, `mobile/src/screens/ExerciseSelectScreen.tsx`, `mobile/src/screens/SetSummaryScreen.tsx`, `mobile/src/screens/components/PositionCheck.tsx`, `mobile/src/screens/components/RestTimer.tsx`
- Test: `mobile/src/session/__tests__/sessionMachine.test.ts`
- Modify: `mobile/App.tsx` (screen switching), `mobile/src/screens/CoachScreen.tsx`

**Interfaces:**
- Consumes: `CoachState`, `visibleJoints` (positioning check needs the exercise's required joints), `ExerciseName`.
- Produces:
  - `type SessionPhase = 'select' | 'positioning' | 'active' | 'rest' | 'summary'`
  - `class SessionMachine` with:
    - `constructor(opts: { exercise: ExerciseName; targetReps: number; totalSets: number; restSeconds: number })`
    - `phase: SessionPhase` (starts `'positioning'`)
    - `onPositionOk(): void` (`positioning → active`)
    - `onRepCounted(reps: number): void` (auto-advances `active → rest` when `reps >= targetReps`, or `→ summary` after the last set)
    - `onRestFinished(): void` (`rest → positioning` for next set)
    - `summary(): { setResults: { attempted: number; good: number }[]; formQualityPct: number }`
    - `recordAttempt(good: boolean): void` — attempted = every counted state-machine cycle; good = counted reps. Form quality % = good/attempted*100 (0 attempts → 0).
  - `PositionCheck` component: given `state` + `exercise`, shows "Step back — full body in frame" until every `visibleJoints(exercise)` landmark has `visibility > 0.5` for 30 consecutive frames, then calls `onPositionOk`.

- [ ] **Step 1: Failing tests for the session machine**

`mobile/src/session/__tests__/sessionMachine.test.ts`:

```ts
import { SessionMachine } from '../sessionMachine';

const make = () => new SessionMachine({ exercise: 'Bicep Curl', targetReps: 3, totalSets: 2, restSeconds: 30 });

test('happy path: positioning → active → rest → positioning → active → summary', () => {
  const m = make();
  expect(m.phase).toBe('positioning');
  m.onPositionOk();
  expect(m.phase).toBe('active');
  [1, 2, 3].forEach((r) => { m.recordAttempt(true); m.onRepCounted(r); });
  expect(m.phase).toBe('rest');
  m.onRestFinished();
  expect(m.phase).toBe('positioning');
  m.onPositionOk();
  [1, 2, 3].forEach((r) => { m.recordAttempt(true); m.onRepCounted(r); });
  expect(m.phase).toBe('summary');
});

test('form quality counts bad attempts', () => {
  const m = make();
  m.onPositionOk();
  m.recordAttempt(true); m.onRepCounted(1);
  m.recordAttempt(false);            // attempted rep, bad form, not counted
  m.recordAttempt(true); m.onRepCounted(2);
  m.recordAttempt(true); m.onRepCounted(3);
  m.onRestFinished(); m.onPositionOk();
  [1, 2, 3].forEach((r) => { m.recordAttempt(true); m.onRepCounted(r); });
  const s = m.summary();
  expect(s.setResults[0]).toEqual({ attempted: 4, good: 3 });
  expect(s.formQualityPct).toBeCloseTo((6 / 7) * 100, 1);
});
```

Run: `cd mobile && npm test -- sessionMachine` — Expected: FAIL.

- [ ] **Step 2: Implement `SessionMachine`**

`mobile/src/session/sessionMachine.ts`:

```ts
import type { ExerciseName } from '../core/exercises';

export type SessionPhase = 'select' | 'positioning' | 'active' | 'rest' | 'summary';

export interface SessionOpts {
  exercise: ExerciseName;
  targetReps: number;
  totalSets: number;
  restSeconds: number;
}

interface SetResult { attempted: number; good: number }

export class SessionMachine {
  phase: SessionPhase = 'positioning';
  currentSet = 1;
  private results: SetResult[] = [{ attempted: 0, good: 0 }];

  constructor(readonly opts: SessionOpts) {}

  onPositionOk(): void {
    if (this.phase === 'positioning') this.phase = 'active';
  }

  recordAttempt(good: boolean): void {
    if (this.phase !== 'active') return;
    const r = this.results[this.results.length - 1];
    r.attempted += 1;
    if (good) r.good += 1;
  }

  onRepCounted(reps: number): void {
    if (this.phase !== 'active' || reps < this.opts.targetReps) return;
    if (this.currentSet >= this.opts.totalSets) {
      this.phase = 'summary';
    } else {
      this.phase = 'rest';
    }
  }

  onRestFinished(): void {
    if (this.phase !== 'rest') return;
    this.currentSet += 1;
    this.results.push({ attempted: 0, good: 0 });
    this.phase = 'positioning';
  }

  summary(): { setResults: SetResult[]; formQualityPct: number } {
    const attempted = this.results.reduce((s, r) => s + r.attempted, 0);
    const good = this.results.reduce((s, r) => s + r.good, 0);
    return {
      setResults: this.results,
      formQualityPct: attempted === 0 ? 0 : (good / attempted) * 100,
    };
  }
}
```

- [ ] **Step 3: Run to verify pass** — `cd mobile && npm test -- sessionMachine` → PASS.

- [ ] **Step 4: Screens and wiring**

- `ExerciseSelectScreen`: a list of the five exercise names (from `Object.keys(EXERCISES)`); tapping one starts a session (default 3 sets × 10 reps, 60 s rest) and navigates to `CoachScreen`.
- `PositionCheck` (overlaid on camera during `positioning`): implements the 30-consecutive-frame visibility rule against `visibleJoints(exercise)`; shows instruction text; calls `onPositionOk`.
- `RestTimer` (overlaid during `rest`): countdown from `restSeconds` via `setInterval`, "Skip" button; both call `onRestFinished`.
- `SetSummaryScreen`: renders `summary()` — per-set attempted/good and overall form quality % with the good/bad color scheme; "Done" returns to select.
- `CoachScreen` drives `SessionMachine`: detect rep-count increases from successive `CoachState.reps` values → `recordAttempt(true)` + `onRepCounted`; detect completed-but-uncounted cycles (stage returned to start without reps incrementing after a `Bad` label) → `recordAttempt(false)`.
- `App.tsx`: simple `useState<SessionPhase | 'select'>`-based switcher (no navigation library needed for Phase 1).

- [ ] **Step 5: Verify on device** — full loop: pick Squat → position → 3 short sets → summary shows form quality %.

- [ ] **Step 6: Commit**

```bash
git add mobile/src/session mobile/src/screens mobile/App.tsx
git commit -m "feat(session): guided session flow with positioning check, rest timer, summary"
```

---

### Task 17: Voice cues (TTS)

**Files:**
- Create: `mobile/src/session/voiceCoach.ts`
- Test: `mobile/src/session/__tests__/voiceCoach.test.ts`
- Modify: `mobile/src/screens/CoachScreen.tsx`

**Interfaces:**
- Consumes: `CoachState`, `ExerciseName`.
- Produces: `class VoiceCoach { constructor(speak: (text: string) => void, cooldownMs?: number); onState(state: CoachState, exercise: ExerciseName, nowMs: number): void }` — speaks rep counts on increment ("Three!") and one correction cue per bad-form episode, rate-limited (default cooldown 4000 ms). `speak` is injected: `expo-speech`'s `Speak` in the app, a spy in tests.

- [ ] **Step 1: Failing test**

`mobile/src/session/__tests__/voiceCoach.test.ts`:

```ts
import { VoiceCoach } from '../voiceCoach';
import type { CoachState } from '../../coach/pipeline';

const s = (over: Partial<CoachState>): CoachState => ({
  reps: 0, stage: 'down', formLabel: 'Good Curl', confidence: 90,
  smoothedLandmarks: [], badJoints: [], ...over,
});

test('announces rep increments', () => {
  const spoken: string[] = [];
  const vc = new VoiceCoach((t) => spoken.push(t));
  vc.onState(s({ reps: 0 }), 'Bicep Curl', 0);
  vc.onState(s({ reps: 1 }), 'Bicep Curl', 100);
  vc.onState(s({ reps: 1 }), 'Bicep Curl', 200);
  vc.onState(s({ reps: 2 }), 'Bicep Curl', 300);
  expect(spoken).toEqual(['1', '2']);
});

test('bad form cue fires once per episode with cooldown', () => {
  const spoken: string[] = [];
  const vc = new VoiceCoach((t) => spoken.push(t), 4000);
  vc.onState(s({ formLabel: 'Bad Curl' }), 'Bicep Curl', 0);
  vc.onState(s({ formLabel: 'Bad Curl' }), 'Bicep Curl', 1000);   // within cooldown
  vc.onState(s({ formLabel: 'Good Curl' }), 'Bicep Curl', 2000);
  vc.onState(s({ formLabel: 'Bad Curl' }), 'Bicep Curl', 5000);   // new episode, past cooldown
  expect(spoken).toEqual(['Keep your elbows in', 'Keep your elbows in']);
});
```

Run: `cd mobile && npm test -- voiceCoach` — Expected: FAIL.

- [ ] **Step 2: Implement**

`mobile/src/session/voiceCoach.ts`:

```ts
import type { CoachState } from '../coach/pipeline';
import type { ExerciseName } from '../core/exercises';

const CUES: Record<ExerciseName, string> = {
  'Bicep Curl': 'Keep your elbows in',
  'Squat': 'Chest up, sit deeper',
  'Lateral Raise': 'Slow down, arms straighter',
  'Shoulder Press': 'Keep your hips under you',
  'Tricep Finisher': 'Lock your upper arm still',
};

export class VoiceCoach {
  private lastReps = 0;
  private lastCueAt = -Infinity;
  private inBadEpisode = false;

  constructor(
    private readonly speak: (text: string) => void,
    private readonly cooldownMs: number = 4000,
  ) {}

  onState(state: CoachState, exercise: ExerciseName, nowMs: number): void {
    if (state.reps > this.lastReps) {
      this.speak(String(state.reps));
      this.lastReps = state.reps;
    }
    const isBad = state.formLabel.includes('Bad');
    if (isBad && !this.inBadEpisode && nowMs - this.lastCueAt >= this.cooldownMs) {
      this.speak(CUES[exercise]);
      this.lastCueAt = nowMs;
    }
    this.inBadEpisode = isBad;
  }
}
```

- [ ] **Step 3: Run to verify pass** — `cd mobile && npm test -- voiceCoach` → PASS.

- [ ] **Step 4: Wire to expo-speech**

```bash
cd mobile && npx expo install expo-speech
```

In `CoachScreen`: `const voice = useRef(new VoiceCoach((t) => Speech.speak(t))).current;` and call `voice.onState(state, exercise, Date.now())` in a `useEffect` on `state`. Add a mute toggle button in the screen header (skip `Speech.speak` when muted).

- [ ] **Step 5: Verify on device** — reps are spoken; deliberate bad form triggers a cue at most every 4 s.

- [ ] **Step 6: Commit**

```bash
git add mobile/src/session mobile/src/screens mobile/package.json mobile/package-lock.json
git commit -m "feat(session): rate-limited voice cues via expo-speech"
```

---

### Task 18: FPS benchmark + resolution fallback

**Files:**
- Create: `mobile/src/coach/fpsMeter.ts`
- Test: `mobile/src/coach/__tests__/fpsMeter.test.ts`
- Modify: `mobile/src/screens/CoachScreen.tsx`

**Interfaces:**
- Produces: `class FpsMeter { tick(nowMs: number): void; fps(): number; }` (rolling 2-second window) and `pickResolution(measuredFps: number): { width: number; height: number }` — returns 480×360 normally, 320×240 when `measuredFps < 15`. CoachScreen shows the FPS in a dev-only badge (`__DEV__`) and applies `pickResolution` after the first 5 seconds of a session.

- [ ] **Step 1: Failing test**

`mobile/src/coach/__tests__/fpsMeter.test.ts`:

```ts
import { FpsMeter, pickResolution } from '../fpsMeter';

test('fps over a rolling window', () => {
  const m = new FpsMeter();
  for (let t = 0; t <= 2000; t += 50) m.tick(t);   // 20 fps
  expect(m.fps()).toBeCloseTo(20, 0);
});

test('resolution fallback below 15fps', () => {
  expect(pickResolution(30)).toEqual({ width: 480, height: 360 });
  expect(pickResolution(14)).toEqual({ width: 320, height: 240 });
});
```

Run: `cd mobile && npm test -- fpsMeter` — Expected: FAIL.

- [ ] **Step 2: Implement**

`mobile/src/coach/fpsMeter.ts`:

```ts
const WINDOW_MS = 2000;

export class FpsMeter {
  private ticks: number[] = [];

  tick(nowMs: number): void {
    this.ticks.push(nowMs);
    const cutoff = nowMs - WINDOW_MS;
    while (this.ticks.length > 0 && this.ticks[0] < cutoff) this.ticks.shift();
  }

  fps(): number {
    if (this.ticks.length < 2) return 0;
    const span = this.ticks[this.ticks.length - 1] - this.ticks[0];
    return span === 0 ? 0 : ((this.ticks.length - 1) * 1000) / span;
  }
}

export function pickResolution(measuredFps: number): { width: number; height: number } {
  return measuredFps < 15 ? { width: 320, height: 240 } : { width: 480, height: 360 };
}
```

- [ ] **Step 3: Run to verify pass** — `cd mobile && npm test -- fpsMeter` → PASS.

- [ ] **Step 4: Wire into CoachScreen** — `meter.tick(Date.now())` inside `onPose`; dev badge shows `meter.fps().toFixed(0)`; after 5 s of `active` phase, if fps < 15 switch the camera `format` request to the fallback resolution (state variable feeding `useCameraFormat`).

- [ ] **Step 5: Record baseline** — run on the available physical devices; note fps in `mobile/README.md` under "Device benchmarks" with device model + date.

- [ ] **Step 6: Commit**

```bash
git add mobile/src/coach mobile/src/screens mobile/README.md
git commit -m "feat(perf): fps meter and resolution fallback below 15fps"
```

---

### Task 19: Beta build config + tracker update

**Files:**
- Create: `mobile/eas.json`
- Modify: `mobile/app.json` (name, slug, bundle identifiers, icon placeholder), `TASKS.md`

**Interfaces:**
- Produces: EAS build profiles for internal distribution; Phase 1 checklist updated.

- [ ] **Step 1: EAS config**

`mobile/eas.json`:

```json
{
  "cli": { "appVersionSource": "remote" },
  "build": {
    "development": {
      "developmentClient": true,
      "distribution": "internal"
    },
    "preview": {
      "distribution": "internal",
      "ios": { "simulator": false }
    }
  }
}
```

In `mobile/app.json`: set `"name": "GymForm AI"`, `"slug": "gymform-ai"`, `ios.bundleIdentifier`/`android.package` `com.gymform.mobile`.

- [ ] **Step 2: Kick off internal builds**

```bash
cd mobile
npx eas build --profile preview --platform android
npx eas build --profile preview --platform ios   # requires Apple Developer account login
```

Expected: builds complete; install on device via the EAS link; full session flow works on the installed build (not just dev client).

- [ ] **Step 3: Update TASKS.md**

Check off every completed Phase 1 item in `TASKS.md` (all boxes under "Phase 1 — CV core", plus "Phase 1 implementation plan" under Phase 0).

- [ ] **Step 4: Commit**

```bash
git add mobile/eas.json mobile/app.json TASKS.md
git commit -m "feat(build): EAS internal-distribution profiles; check off Phase 1"
```

---

## Deviations & risk notes for the implementer

- **VisionCamera / MediaPipe API drift:** the frame-processor plugin registration API (`FrameProcessorPluginRegistry`, `VISION_EXPORT_SWIFT_FRAME_PROCESSOR`) is correct for VisionCamera v4 at plan time — if signatures differ on the installed version, follow the official "Creating Frame Processor Plugins" guide for the installed version and keep the **`detectPose` contract** (`null | { landmarks: [{x,y,z,visibility}×33] }`) unchanged; everything downstream depends only on that.
- **Orientation/mirroring:** normalized landmark coordinates may need x-mirroring (front camera) and rotation fixes per platform. Fix at the overlay/`pipeline` input boundary, verify visually; do not change core math.
- **Performance:** if `toBitmap()` conversion on Android is too slow, switch the plugin to MediaPipe's `RunningMode.LIVE_STREAM` with a YUV→MPImage path; the JS contract is unchanged.
- **Anything ambiguous:** the Python code in the repo root is the reference. When in doubt, do what `app.py` does.
