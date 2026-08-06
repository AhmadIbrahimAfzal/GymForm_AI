const WINDOW_MS = 2000;

export class FpsMeter {
  private ticks: number[] = [];

  tick(nowMs: number): void {
    this.ticks.push(nowMs);
    const cutoff = nowMs - WINDOW_MS;
    while (this.ticks.length > 0 && this.ticks[0] < cutoff) {
      this.ticks.shift();
    }
  }

  fps(): number {
    if (this.ticks.length < 2) return 0;
    const span = this.ticks[this.ticks.length - 1] - this.ticks[0];
    return span === 0 ? 0 : ((this.ticks.length - 1) * 1000) / span;
  }
}

export function pickResolution(measuredFps: number): { width: number; height: number } {
  return measuredFps < 15 ? { width: 320, height: 240 } : { width: 480, height: 360 };
}
