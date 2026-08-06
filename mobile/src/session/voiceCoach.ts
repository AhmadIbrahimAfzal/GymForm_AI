import type { CoachState } from '../coach/pipeline';
import type { ExerciseName } from '../core/exercises';

const CUES: Record<ExerciseName, string> = {
  'Bicep Curl': 'Keep your elbows in',
  'Squat': 'Chest up, sit deeper',
  'Lateral Raise': 'Slow down, arms straighter',
  'Shoulder Press': 'Keep your hips under you',
  'Tricep Finisher': 'Lock your upper arm still',
};

export class VoiceCoach {
  private lastReps = 0;
  private lastCueAt = -Infinity;
  private inBadEpisode = false;

  constructor(
    private readonly speak: (text: string) => void,
    private readonly cooldownMs: number = 4000,
  ) {}

  reset(): void {
    this.lastReps = 0;
    this.lastCueAt = -Infinity;
    this.inBadEpisode = false;
  }

  onState(state: CoachState, exercise: ExerciseName, nowMs: number): void {
    if (state.reps > this.lastReps) {
      this.speak(String(state.reps));
      this.lastReps = state.reps;
    }
    const isBad = state.formLabel.includes('Bad');
    if (isBad && !this.inBadEpisode && nowMs - this.lastCueAt >= this.cooldownMs) {
      this.speak(CUES[exercise]);
      this.lastCueAt = nowMs;
    }
    this.inBadEpisode = isBad;
  }
}
