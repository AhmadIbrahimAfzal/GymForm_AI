import { computeJointAngles } from '../core/angles';
import { EmaFilter } from '../core/ema';
import { MajorityVote } from '../core/majorityVote';
import { anglesToInput, classify } from '../core/classifier';
import { EXERCISES, type Exercise, type ExerciseName, type ExerciseAngles } from '../core/exercises';
import type { OnnxClassifier } from '../ml/onnxClassifier';

export interface PoseResult {
  landmarks: { x: number; y: number; z: number; visibility: number }[];
}

export interface CoachState {
  reps: number;
  stage: string;
  formLabel: string;
  confidence: number;
  smoothedLandmarks: number[][] | null;
  badJoints: number[];
}

const CONFIDENCE_FLOOR = 60;

export class CoachPipeline {
  private ema = new EmaFilter(0.5);
  private vote = new MajorityVote(10);
  private machine: Exercise;
  private state: CoachState = {
    reps: 0,
    stage: 'down',
    formLabel: 'Waiting...',
    confidence: 0,
    smoothedLandmarks: null,
    badJoints: [],
  };

  constructor(private exercise: ExerciseName, private classifier: OnnxClassifier) {
    this.machine = EXERCISES[exercise]();
  }

  setExercise(e: ExerciseName): void {
    this.exercise = e;
    this.machine = EXERCISES[e]();
    this.ema.reset();
    this.vote.reset();
    this.state = {
      reps: 0,
      stage: this.machine.stage,
      formLabel: 'Waiting...',
      confidence: 0,
      smoothedLandmarks: null,
      badJoints: [],
    };
  }

  async process(pose: PoseResult | null): Promise<CoachState> {
    if (pose === null || !pose.landmarks || pose.landmarks.length < 33) {
      return { ...this.state, smoothedLandmarks: null };
    }

    const raw = pose.landmarks.map((lm) => [lm.x, lm.y, lm.z]);
    const smoothed = this.ema.apply(raw);
    const angles = computeJointAngles(smoothed);

    const exAngles: ExerciseAngles = {
      ...angles,
      active_arm:
        (pose.landmarks[13]?.visibility ?? 0) > (pose.landmarks[14]?.visibility ?? 0)
          ? 'left'
          : 'right',
    };

    const input = anglesToInput(angles, this.exercise);
    const logits = await this.classifier.run(input);
    const { label, confidence } = classify(logits, this.exercise);

    let smoothedLabel = this.vote.add(label);
    if (confidence < CONFIDENCE_FLOOR) {
      smoothedLabel = 'Tracking...';
    }

    const [reps, stage] = this.machine.update(exAngles, smoothedLabel);
    const badJoints = smoothedLabel.includes('Bad') ? this.badJointsFor(exAngles) : [];

    this.state = {
      reps,
      stage,
      formLabel: smoothedLabel,
      confidence,
      smoothedLandmarks: smoothed,
      badJoints,
    };
    return this.state;
  }

  // Joint indices to highlight red on bad form (condensed from app.py)
  private badJointsFor(a: ExerciseAngles): number[] {
    switch (this.exercise) {
      case 'Bicep Curl':
        return (a.l_shoulder ?? 0) > 40 || (a.r_shoulder ?? 0) > 40
          ? [11, 12, 13, 14, 23, 24]
          : [11, 12, 13, 14, 15, 16];
      case 'Squat':
        return (a.l_hip ?? 180) < 70 || (a.r_hip ?? 180) < 70
          ? [11, 12, 23, 24]
          : [23, 24, 25, 26, 27, 28];
      case 'Lateral Raise':
        return (a.l_elbow ?? 180) < 140 || (a.r_elbow ?? 180) < 140
          ? [11, 12, 13, 14, 15, 16]
          : [11, 12, 13, 14];
      case 'Shoulder Press':
        return (a.l_hip ?? 180) < 160 || (a.r_hip ?? 180) < 160
          ? [11, 12, 23, 24]
          : [11, 12, 13, 14, 15, 16];
      case 'Tricep Finisher':
        return (a.l_shoulder ?? 0) > 45 && (a.r_shoulder ?? 0) > 45
          ? [11, 12, 13, 14]
          : [11, 12, 13, 14, 15, 16];
    }
  }
}
