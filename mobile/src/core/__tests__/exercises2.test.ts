import { EXERCISES, ExerciseName } from '../exercises';
import fx from '../__fixtures__/exercises.json';
import type { ExerciseAngles } from '../exercises/types';

type Step = { angles: ExerciseAngles; label: string; reps: number; stage: string };
const names: ExerciseName[] = ['Lateral Raise', 'Shoulder Press', 'Tricep Finisher'];

test.each(names)('%s matches Python reference', (name) => {
  const machine = EXERCISES[name]();
  for (const s of (fx as Record<string, Step[]>)[name]) {
    const [reps, stage] = machine.update(s.angles, s.label);
    expect(reps).toBe(s.reps);
    expect(stage).toBe(s.stage);
  }
});

test('registry has all five exercises', () => {
  expect(Object.keys(EXERCISES).sort()).toEqual(
    ['Bicep Curl', 'Lateral Raise', 'Shoulder Press', 'Squat', 'Tricep Finisher']);
});
