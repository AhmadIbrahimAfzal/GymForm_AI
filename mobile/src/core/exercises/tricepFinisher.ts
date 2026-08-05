import type { Exercise, ExerciseAngles, Stage } from './types';

export class TricepFinisher implements Exercise {
  repCount = 0;
  stage: Stage = 'bent';
  private formWasBad = false;

  update(angles: ExerciseAngles, smoothedClass: string): [number, Stage] {
    const arm = angles.active_arm ?? 'left';
    const el = (arm === 'left' ? angles.l_elbow : angles.r_elbow) ?? 180;
    if (smoothedClass.includes('Bad')) this.formWasBad = true;
    if (el < 70) {
      this.stage = 'bent';
      this.formWasBad = false;
    }
    if (el > 150 && this.stage === 'bent') {
      this.stage = 'straight';
      if (!this.formWasBad && smoothedClass.includes('Good')) this.repCount++;
    }
    return [this.repCount, this.stage];
  }
}
