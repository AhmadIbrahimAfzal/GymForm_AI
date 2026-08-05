import * as ortNode from 'onnxruntime-node';
import * as path from 'path';
import { createOnnxClassifier } from '../onnxClassifier';
import { classify, anglesToInput } from '../../core/classifier';
import fx from '../../core/__fixtures__/classifier.json';
import type { ExerciseName } from '../../core/exercises';
import type { JointAngles } from '../../core/angles';

type Verdict = { label: string; confidence: number; logits?: number[] };
type Row = {
  angles: number[];
  logits: number[];
  verdicts: Record<string, Verdict>;
};

const MODEL_PATH = path.resolve(__dirname, '../../../../mobile/assets/models/gym_model_v2.onnx');

test('ONNX model loads and runs inference using onnxruntime-node', async () => {
  const clf = await createOnnxClassifier(MODEL_PATH, ortNode);
  const input = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0]);
  const logits = await clf.run(input);
  expect(logits.length).toBe(10);
});

test('ONNX logits and end-to-end verdicts match the PyTorch reference for fixture samples', async () => {
  const clf = await createOnnxClassifier(MODEL_PATH, ortNode);
  const rows = (fx as Row[]).slice(0, 25);

  for (const row of rows) {
    const rawAngles: JointAngles = {
      l_elbow: row.angles[0],
      r_elbow: row.angles[1],
      l_shoulder: row.angles[2],
      r_shoulder: row.angles[3],
      l_hip: row.angles[4],
      r_hip: row.angles[5],
      l_knee: row.angles[6],
      r_knee: row.angles[7],
    };

    for (const [ex, expectedVerdict] of Object.entries(row.verdicts)) {
      const input = anglesToInput(rawAngles, ex as ExerciseName);
      const exLogits = await clf.run(input);

      // Check logits match expected PyTorch logits for this exercise
      if (expectedVerdict.logits) {
        expectedVerdict.logits.forEach((exp, i) => {
          expect(exLogits[i]).toBeCloseTo(exp, 3);
        });
      }

      const verdict = classify(exLogits, ex as ExerciseName);
      expect(verdict.label).toBe(expectedVerdict.label);
      expect(verdict.confidence).toBeCloseTo(expectedVerdict.confidence, 3);
    }
  }
});
