import { CoachPipeline, type PoseResult } from '../pipeline';
import type { OnnxClassifier } from '../../ml/onnxClassifier';

// Mock classifier returning high "Good Curl" logit
const goodCurlClassifier: OnnxClassifier = {
  run: async () => Float32Array.from([0, 10, 0, 0, 0, 0, 0, 0, 0, 0]),
};

function poseWithElbowAngle(deg: number): PoseResult {
  const lms = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 1 }));
  // Shoulder (11/12) at (0.5, 0.3), Elbow (13/14) at (0.5, 0.5)
  lms[11] = { x: 0.5, y: 0.3, z: 0, visibility: 1 };
  lms[12] = { x: 0.7, y: 0.3, z: 0, visibility: 1 };
  lms[13] = { x: 0.5, y: 0.5, z: 0, visibility: 1 };
  lms[14] = { x: 0.7, y: 0.5, z: 0, visibility: 1 };

  // Extended arm (deg > 120): wrist below elbow at y=0.7 (elbow angle = 180)
  // Curled arm (deg < 90): wrist above elbow near shoulder at y=0.3 (elbow angle = 0)
  const wristY = deg > 120 ? 0.7 : 0.3;
  lms[15] = { x: 0.5, y: wristY, z: 0, visibility: 1 };
  lms[16] = { x: 0.7, y: wristY, z: 0, visibility: 1 };
  return { landmarks: lms };
}

test('a full good curl counts one rep end-to-end', async () => {
  const p = new CoachPipeline('Bicep Curl', goodCurlClassifier);
  for (let i = 0; i < 15; i++) await p.process(poseWithElbowAngle(170));
  let s;
  for (let i = 0; i < 15; i++) s = await p.process(poseWithElbowAngle(30));
  expect(s!.formLabel).toBe('Good Curl');
  expect(s!.reps).toBe(1);
  expect(s!.stage).toBe('up');
});

test('null pose yields null landmarks and preserved state', async () => {
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
