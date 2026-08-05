import { EmaFilter } from '../ema';
import { MajorityVote } from '../majorityVote';
import emaFx from '../__fixtures__/ema.json';
import mvFx from '../__fixtures__/majority_vote.json';

test('EmaFilter matches Python reference', () => {
  const f = new EmaFilter(emaFx.alpha);
  (emaFx.frames as number[][][]).forEach((frame, i) => {
    const out = f.apply(frame);
    const exp = (emaFx.expected as number[][][])[i];
    out.forEach((lm, j) => lm.forEach((v, k) => expect(v).toBeCloseTo(exp[j][k], 6)));
  });
});

test('EmaFilter.reset restarts from next frame', () => {
  const f = new EmaFilter(0.5);
  f.apply([[0, 0, 0]]);
  f.reset();
  expect(f.apply([[1, 1, 1]])).toEqual([[1, 1, 1]]);
});

test('MajorityVote matches Python Counter behavior', () => {
  const mv = new MajorityVote(mvFx.maxlen);
  (mvFx.sequence as string[]).forEach((label, i) => {
    expect(mv.add(label)).toBe((mvFx.expected as string[])[i]);
  });
});
