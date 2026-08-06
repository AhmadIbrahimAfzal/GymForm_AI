import type { ExerciseName } from '../core/exercises';

export type SessionPhase = 'select' | 'positioning' | 'active' | 'rest' | 'summary';

export interface SessionOpts {
  exercise: ExerciseName;
  targetReps: number;
  totalSets: number;
  restSeconds: number;
}

export interface SetResult {
  attempted: number;
  good: number;
}

export class SessionMachine {
  phase: SessionPhase = 'positioning';
  currentSet = 1;
  private results: SetResult[] = [{ attempted: 0, good: 0 }];

  constructor(readonly opts: SessionOpts) {}

  onPositionOk(): void {
    if (this.phase === 'positioning') {
      this.phase = 'active';
    }
  }

  recordAttempt(good: boolean): void {
    if (this.phase !== 'active') return;
    const r = this.results[this.results.length - 1];
    r.attempted += 1;
    if (good) r.good += 1;
  }

  onRepCounted(reps: number): void {
    if (this.phase !== 'active' || reps < this.opts.targetReps) return;
    if (this.currentSet >= this.opts.totalSets) {
      this.phase = 'summary';
    } else {
      this.phase = 'rest';
    }
  }

  onRestFinished(): void {
    if (this.phase !== 'rest') return;
    this.currentSet += 1;
    this.results.push({ attempted: 0, good: 0 });
    this.phase = 'positioning';
  }

  summary(): { setResults: SetResult[]; formQualityPct: number } {
    const attempted = this.results.reduce((s, r) => s + r.attempted, 0);
    const good = this.results.reduce((s, r) => s + r.good, 0);
    return {
      setResults: this.results,
      formQualityPct: attempted === 0 ? 0 : (good / attempted) * 100,
    };
  }
}
