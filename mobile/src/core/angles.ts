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
