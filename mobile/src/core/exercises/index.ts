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
