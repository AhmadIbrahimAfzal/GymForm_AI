import type { JointAngles } from '../angles';

export type Stage = 'up' | 'down' | 'bent' | 'straight';
export type ExerciseAngles = Partial<JointAngles> & { active_arm?: 'left' | 'right' };

export interface Exercise {
  repCount: number;
  stage: Stage;
  update(angles: ExerciseAngles, smoothedClass: string): [number, Stage];
}
