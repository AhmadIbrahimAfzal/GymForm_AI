import { BicepCurl } from '../exercises/bicepCurl';
import { Squat } from '../exercises/squat';
import fx from '../__fixtures__/exercises.json';
import type { Exercise, ExerciseAngles } from '../exercises/types';

type Step = { angles: ExerciseAngles; label: string; reps: number; stage: string };

function replay(machine: Exercise, steps: Step[]) {
  for (const s of steps) {
    const [reps, stage] = machine.update(s.angles, s.label);
    expect(reps).toBe(s.reps);
    expect(stage).toBe(s.stage);
  }
}

test('BicepCurl matches Python reference', () => {
  replay(new BicepCurl(), (fx as Record<string, Step[]>)['Bicep Curl']);
});

test('Squat matches Python reference', () => {
  replay(new Squat(), (fx as Record<string, Step[]>)['Squat']);
});
