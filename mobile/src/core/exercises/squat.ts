import type { Exercise, ExerciseAngles, Stage } from './types';

export class Squat implements Exercise {
  repCount = 0;
  stage: Stage = 'up';
  private formWasBad = false;

  update(angles: ExerciseAngles, smoothedClass: string): [number, Stage] {
    const lk = angles.l_knee ?? 180;
    const rk = angles.r_knee ?? 180;
    if (smoothedClass.includes('Bad')) this.formWasBad = true;
    if (lk > 140 && rk > 140) {
      this.stage = 'up';
      this.formWasBad = false;
    }
    if (lk < 115 && rk < 115 && this.stage === 'up') {
      this.stage = 'down';
      if (!this.formWasBad && smoothedClass.includes('Good')) this.repCount++;
    }
    return [this.repCount, this.stage];
  }
}
