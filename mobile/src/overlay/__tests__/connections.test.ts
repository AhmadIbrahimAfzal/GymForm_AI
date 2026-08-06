import { POSE_CONNECTIONS, visibleConnections, visibleJoints } from '../connections';

test('face connections are always excluded', () => {
  for (const [s, e] of visibleConnections('Squat')) {
    expect(s).toBeGreaterThan(10);
    expect(e).toBeGreaterThan(10);
  }
});

test('upper-body exercises exclude legs (>=25)', () => {
  for (const [s, e] of visibleConnections('Bicep Curl')) {
    expect(s).toBeLessThan(25);
    expect(e).toBeLessThan(25);
  }
  expect(visibleJoints('Bicep Curl').every((i) => i > 10 && i < 25)).toBe(true);
});

test('squat excludes arms/hands (13-22)', () => {
  for (const [s, e] of visibleConnections('Squat')) {
    expect(s < 13 || s > 22).toBe(true);
    expect(e < 13 || e > 22).toBe(true);
  }
});

test('POSE_CONNECTIONS matches the reference count', () => {
  expect(POSE_CONNECTIONS.length).toBe(35);
});
