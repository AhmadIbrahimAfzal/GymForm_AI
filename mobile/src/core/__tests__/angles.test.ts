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
