import type { Exercise, ExerciseAngles, Stage } from './types';

export class ShoulderPress implements Exercise {
  repCount = 0;
  stage: Stage = 'down';
  private formWasBad = false;

  update(angles: ExerciseAngles, smoothedClass: string): [number, Stage] {
    const le = angles.l_elbow ?? 180;
    const re = angles.r_elbow ?? 180;
    if (smoothedClass.includes('Bad')) this.formWasBad = true;
    if (le < 100 && re < 100) {
      this.stage = 'down';
      this.formWasBad = false;
    }
    if (le > 150 && re > 150 && this.stage === 'down') {
      this.stage = 'up';
      if (!this.formWasBad && smoothedClass.includes('Good')) this.repCount++;
    }
    return [this.repCount, this.stage];
  }
}
