import type { JointAngles } from './angles';
import type { ExerciseName } from './exercises';
import scalerParams from '../../assets/models/scaler_params.json';

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

// StandardScaler normalization: (value - mean) / std
// These params were saved by train_model_v2.py during training.
const SCALER_MEAN = Float32Array.from(scalerParams.mean);
const SCALER_STD = Float32Array.from(scalerParams.std);

function normalize(raw: Float32Array): Float32Array {
  const out = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    out[i] = (raw[i] - SCALER_MEAN[i]) / SCALER_STD[i];
  }
  return out;
}

export function anglesToInput(a: JointAngles, exercise: ExerciseName): Float32Array {
  const v = { ...a };
  if (UPPER_BODY.includes(exercise)) {
    v.l_hip = v.r_hip = v.l_knee = v.r_knee = 180;
  } else {
    v.l_elbow = v.r_elbow = v.l_shoulder = v.r_shoulder = 180;
  }
  const raw = Float32Array.from([
    v.l_elbow, v.r_elbow, v.l_shoulder, v.r_shoulder, v.l_hip, v.r_hip, v.l_knee, v.r_knee,
  ]);
  return normalize(raw);
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
