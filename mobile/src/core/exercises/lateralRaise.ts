import type { Exercise, ExerciseAngles, Stage } from './types';

export class LateralRaise implements Exercise {
  repCount = 0;
  stage: Stage = 'down';
  private formWasBad = false;

  update(angles: ExerciseAngles, smoothedClass: string): [number, Stage] {
    const ls = angles.l_shoulder ?? 0;
    const rs = angles.r_shoulder ?? 0;
    if (smoothedClass.includes('Bad')) this.formWasBad = true;
    if (ls > 65 && rs > 65) this.stage = 'up';
    if (ls < 45 && rs < 45) {
      if (this.stage === 'up') {
        this.stage = 'down';
        if (!this.formWasBad && smoothedClass.includes('Good')) this.repCount++;
      } else {
        this.stage = 'down';
        this.formWasBad = false;
      }
    }
    return [this.repCount, this.stage];
  }
}
