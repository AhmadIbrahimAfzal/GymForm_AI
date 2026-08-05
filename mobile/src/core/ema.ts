export class EmaFilter {
  private state: number[][] | null = null;
  constructor(private readonly alpha: number = 0.5) {}

  apply(landmarks: number[][]): number[][] {
    if (this.state === null) {
      this.state = landmarks.map((lm) => [...lm]);
    } else {
      for (let i = 0; i < landmarks.length; i++) {
        for (let k = 0; k < landmarks[i].length; k++) {
          this.state[i][k] = this.alpha * landmarks[i][k] + (1 - this.alpha) * this.state[i][k];
        }
      }
    }
    return this.state.map((lm) => [...lm]);
  }

  reset(): void {
    this.state = null;
  }
}
