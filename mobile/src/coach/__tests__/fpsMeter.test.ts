import { FpsMeter, pickResolution } from '../fpsMeter';

test('fps over a rolling window', () => {
  const m = new FpsMeter();
  for (let t = 0; t <= 2000; t += 50) m.tick(t); // 20 fps
  expect(m.fps()).toBeCloseTo(20, 0);
});

test('resolution fallback below 15fps', () => {
  expect(pickResolution(30)).toEqual({ width: 480, height: 360 });
  expect(pickResolution(14)).toEqual({ width: 320, height: 240 });
});
