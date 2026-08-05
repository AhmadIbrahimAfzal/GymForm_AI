import { classify, anglesToInput, LABELS } from '../classifier';
import fx from '../__fixtures__/classifier.json';
import type { ExerciseName } from '../exercises';

type Verdict = { label: string; confidence: number; logits?: number[] };
type Row = {
  angles: number[];
  logits: number[];
  verdicts: Record<string, Verdict>;
};

test('classify matches Python masked softmax for every exercise', () => {
  for (const row of fx as Row[]) {
    for (const [ex, expected] of Object.entries(row.verdicts)) {
      const logits = expected.logits ?? row.logits;
      const got = classify(logits, ex as ExerciseName);
      expect(got.label).toBe(expected.label);
      expect(got.confidence).toBeCloseTo(expected.confidence, 3);
    }
  }
});

test('anglesToInput normalizes and zeroes the irrelevant group to 180', () => {
  const a = {
    l_elbow: 10,
    r_elbow: 20,
    l_shoulder: 30,
    r_shoulder: 40,
    l_hip: 50,
    r_hip: 60,
    l_knee: 70,
    r_knee: 80,
  };

  const curlInput = anglesToInput(a, 'Bicep Curl');
  expect(curlInput.length).toBe(8);

  const squatInput = anglesToInput(a, 'Squat');
  expect(squatInput.length).toBe(8);

  expect(curlInput[0]).not.toBeCloseTo(squatInput[0], 2);
});

test('LABELS ordering matches the model head', () => {
  expect(LABELS[1]).toBe('Good Curl');
  expect(LABELS[8]).toBe('Bad Tricep');
});
