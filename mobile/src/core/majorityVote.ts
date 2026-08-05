export class MajorityVote {
  private history: string[] = [];
  constructor(private readonly maxlen: number = 10) {}

  add(label: string): string {
    this.history.push(label);
    if (this.history.length > this.maxlen) this.history.shift();
    const counts = new Map<string, number>();
    for (const l of this.history) counts.set(l, (counts.get(l) ?? 0) + 1);
    let best = this.history[0], bestCount = 0;
    for (const [l, c] of counts) {           // Map preserves first-seen order,
      if (c > bestCount) { best = l; bestCount = c; }  // matching Counter.most_common ties
    }
    return best;
  }

  reset(): void {
    this.history = [];
  }
}
