import { SessionMachine } from '../sessionMachine';

const make = () =>
  new SessionMachine({ exercise: 'Bicep Curl', targetReps: 3, totalSets: 2, restSeconds: 30 });

test('happy path: positioning -> active -> rest -> positioning -> active -> summary', () => {
  const m = make();
  expect(m.phase).toBe('positioning');
  m.onPositionOk();
  expect(m.phase).toBe('active');
  [1, 2, 3].forEach((r) => {
    m.recordAttempt(true);
    m.onRepCounted(r);
  });
  expect(m.phase).toBe('rest');
  m.onRestFinished();
  expect(m.phase).toBe('positioning');
  m.onPositionOk();
  [1, 2, 3].forEach((r) => {
    m.recordAttempt(true);
    m.onRepCounted(r);
  });
  expect(m.phase).toBe('summary');
});

test('form quality counts bad attempts', () => {
  const m = make();
  m.onPositionOk();
  m.recordAttempt(true);
  m.onRepCounted(1);
  m.recordAttempt(false); // bad form attempt
  m.recordAttempt(true);
  m.onRepCounted(2);
  m.recordAttempt(true);
  m.onRepCounted(3);
  m.onRestFinished();
  m.onPositionOk();
  [1, 2, 3].forEach((r) => {
    m.recordAttempt(true);
    m.onRepCounted(r);
  });
  const s = m.summary();
  expect(s.setResults[0]).toEqual({ attempted: 4, good: 3 });
  expect(s.formQualityPct).toBeCloseTo((6 / 7) * 100, 1);
});
