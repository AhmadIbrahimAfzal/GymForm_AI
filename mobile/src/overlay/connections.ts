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
  if (i <= 10) return false; // Exclude face landmarks
  if (UPPER_BODY.includes(exercise) && i >= 25) return false; // Upper body excludes legs
  if (exercise === 'Squat' && i >= 13 && i <= 22) return false; // Squat excludes arms
  return true;
}

export function visibleConnections(exercise: ExerciseName): [number, number][] {
  return POSE_CONNECTIONS.filter(([s, e]) => jointVisible(s, exercise) && jointVisible(e, exercise));
}

export function visibleJoints(exercise: ExerciseName): number[] {
  return Array.from({ length: 33 }, (_, i) => i).filter((i) => jointVisible(i, exercise));
}
