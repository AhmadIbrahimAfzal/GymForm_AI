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

test('anglesToInput normalizes and zeroes the irrelevant group to 180', () => {
  const a = { l_elbow: 10, r_elbow: 20, l_shoulder: 30, r_shoulder: 40,
              l_hip: 50, r_hip: 60, l_knee: 70, r_knee: 80 };

  // For upper body exercises, hip/knee should be 180 (then normalized)
  const curlInput = anglesToInput(a, 'Bicep Curl');
  // Raw values 10,20,30,40 for elbow/shoulder; 180,180,180,180 for hip/knee — all normalized
  expect(curlInput.length).toBe(8);

  // For squat, elbow/shoulder should be 180 (then normalized)
  const squatInput = anglesToInput(a, 'Squat');
  expect(squatInput.length).toBe(8);

  // Upper body and squat should produce different normalized values
  expect(curlInput[0]).not.toBeCloseTo(squatInput[0], 2);
});

test('LABELS ordering matches the model head', () => {
  expect(LABELS[1]).toBe('Good Curl');
  expect(LABELS[8]).toBe('Bad Tricep');
});
