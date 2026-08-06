import { VoiceCoach } from '../voiceCoach';
import type { CoachState } from '../../coach/pipeline';

const s = (over: Partial<CoachState>): CoachState => ({
  reps: 0,
  stage: 'down',
  formLabel: 'Good Curl',
  confidence: 90,
  smoothedLandmarks: [],
  badJoints: [],
  ...over,
});

test('announces rep increments', () => {
  const spoken: string[] = [];
  const vc = new VoiceCoach((t) => spoken.push(t));
  vc.onState(s({ reps: 0 }), 'Bicep Curl', 0);
  vc.onState(s({ reps: 1 }), 'Bicep Curl', 100);
  vc.onState(s({ reps: 1 }), 'Bicep Curl', 200);
  vc.onState(s({ reps: 2 }), 'Bicep Curl', 300);
  expect(spoken).toEqual(['1', '2']);
});

test('bad form cue fires once per episode with cooldown', () => {
  const spoken: string[] = [];
  const vc = new VoiceCoach((t) => spoken.push(t), 4000);
  vc.onState(s({ formLabel: 'Bad Curl' }), 'Bicep Curl', 0);
  vc.onState(s({ formLabel: 'Bad Curl' }), 'Bicep Curl', 1000); // within cooldown
  vc.onState(s({ formLabel: 'Good Curl' }), 'Bicep Curl', 2000);
  vc.onState(s({ formLabel: 'Bad Curl' }), 'Bicep Curl', 5000); // new episode, past cooldown
  expect(spoken).toEqual(['Keep your elbows in', 'Keep your elbows in']);
});
