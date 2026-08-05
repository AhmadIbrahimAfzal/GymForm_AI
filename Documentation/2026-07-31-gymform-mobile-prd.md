# GymForm AI Mobile — Product Requirements Document

**Date:** 2026-07-31
**Status:** Approved for design
**Companion doc:** [2026-07-31-gymform-mobile-design.md](2026-07-31-gymform-mobile-design.md)

## 1. Vision

GymForm AI Mobile is a freemium fitness app for iOS and Android that acts as a real-time AI form coach. Using only the phone's camera, it tracks the user's body during exercises, counts strict reps (reps only count when form is good), and gives live visual and voice feedback when form breaks down. Around this core it builds a full product: workout history, AI-personalized programs, streaks, and social accountability.

**Differentiator:** every workout produces *form-quality data* — good reps vs. attempted reps per set — which no mainstream fitness tracker has. All computer vision runs on-device; no video ever leaves the phone.

## 2. Problem

- Beginners don't know if their form is right, and bad form causes injury and slow progress.
- Personal trainers are expensive; watching yourself in a mirror mid-set doesn't work.
- Existing fitness apps track *that* you worked out, not *how well* you performed each rep.

## 3. Goals & success metrics

| Goal | Metric | Target (12 months post-launch) |
|---|---|---|
| Prove the coach works on real phones | Live coach sustains ≥15 fps pose tracking on mid-range devices | ≥95% of supported devices |
| Acquisition | App store installs | 50k installs |
| Activation | Users completing a first coached set within first session | ≥60% |
| Retention | Week-4 retention | ≥20% |
| Revenue | Free → premium conversion | ≥3% of monthly actives |
| Trust | No video/landmark data leaves device | 0 exceptions, ever |

## 4. Target users

- **Primary — gym beginners (18–35):** self-conscious about form, train alone, phone always with them.
- **Secondary — home lifters:** dumbbell/bodyweight training, no trainer access, need structure and accountability.
- **Tertiary — returning lifters:** rebuilding after a break or injury; care about controlled, safe progression.

## 5. Product scope

### 5.1 Live Coach (core, free with limits)

- Camera-based pose tracking with skeleton overlay: green when form is good, offending joints highlighted red when it is not.
- Exercise-specific rep counting via state machines; strict counting — a rep with bad form at any point does not count.
- Live stats: reps, stage (up/down), form verdict with confidence, and per-set **form quality %** (good reps ÷ attempted reps) — the app's signature metric.
- Voice feedback (TTS) for correction cues ("keep your elbows in") since users can't watch the screen mid-set.
- Guided session flow: pick exercise → positioning check ("full body visible") → coached set → rest timer → set summary.
- Launch exercises (5): Bicep Curl, Squat, Lateral Raise, Shoulder Press, Tricep Finisher.
- Works fully offline.

### 5.2 Workout tracking

- Every session stored locally first (offline-first): sets, reps, good-rep count, form quality %, duration, date.
- Progress charts: volume over time, form quality trend per exercise, personal records (best good-rep set).
- History browsable by calendar and by exercise.

### 5.3 Accounts & sync

- App fully usable anonymously; account (Apple / Google / email sign-in) required only for cloud sync and social.
- On sign-up, local history merges to the cloud. Sync is background and non-blocking.

### 5.4 Programs (premium)

- **Authored programs:** structured multi-week routines built from supported exercises (e.g. "Beginner Full Body, 3×/week, 4 weeks"), run as guided sessions by the Live Coach.
- **AI-tailored programs:** onboarding quiz (goal, experience, days/week, session length, limitations) → LLM generates a personalized program constrained to supported exercises. Regenerated at program boundaries (~every 4 weeks) or on goal change.
- **Adaptive progression:** a deterministic weekly engine adjusts sets/reps from actual performance and form-quality data, with explainable changes ("+2 reps because your form was 92% last week"). Runs offline.

### 5.5 Gamification & notifications

- Workout-day streak, weekly session goal with progress ring, personal records, small badge set (first workout, 7-day streak, 100 good reps per exercise, program completion).
- Push notifications: workout reminder at user-chosen time, streak-about-to-break warning (only if streak ≥3), program-day nudge. Max one notification/day; each type individually toggleable.

### 5.6 Social (minimal v1)

- Add friends by username or invite link.
- One weekly leaderboard among friends: total good reps.
- Shareable session-summary card (image only — no video, reinforcing privacy).
- Explicit non-goals: no feed, no comments, no public profiles.

## 6. Monetization

**Model:** freemium subscription (monthly + annual with 7-day trial), via RevenueCat.

| | Free | Premium |
|---|---|---|
| Exercises | Bicep Curl, Squat | All 5 (+ future) |
| Live coach | Full experience | Full experience |
| History | Last 7 days | Unlimited + charts |
| Streaks & goals | ✓ | ✓ |
| Programs (authored + AI) | — | ✓ |
| Social | ✓ | ✓ |

Paywall placement: tapping a locked exercise, opening a program, or viewing history beyond day 7. Never shown mid-workout.

## 7. Platforms & constraints

- iOS and Android, single React Native codebase.
- All ML on-device (latency + privacy). Minimum device bar: sustains ~15 fps pose tracking; below that, camera resolution is reduced before model quality.
- Offline-first: everything except social and program generation works without connectivity.
- Privacy: camera frames and pose landmarks never leave the device; only aggregate workout stats sync.

## 8. Release phases

1. **CV core** — live coach, 5 exercises, guided sessions, TTS. Beta via TestFlight / Play internal track.
2. **Tracking** — local history, PRs, charts.
3. **Accounts & sync** — auth, cloud sync, streaks, push notifications.
4. **Monetization + programs** — paywall, authored programs, AI generation, adaptive engine.
5. **Social** — friends, leaderboard, share cards. Public launch.

## 9. Risks

| Risk | Mitigation |
|---|---|
| CV performance inadequate on low-end Android | Phase 1 first; device matrix benchmarks; resolution fallback; published minimum-device bar |
| Classifier trained on limited data misjudges real users | Form-quality thresholds tuned conservatively (<60% confidence → "Tracking…", no penalty); beta feedback loop; retraining pipeline retained from Python project |
| Only 5 exercises limits perceived value | Positioning: depth of coaching over breadth; exercise roadmap as premium driver |
| Subscription conversion below target | Paywall experiments via RevenueCat; AI programs as the premium anchor |
| App store rejection (health claims, camera use) | No medical/injury-prevention claims; clear camera-purpose strings; privacy-first messaging |

## 10. Out of scope (v1)

Wearable integration, nutrition tracking, video recording/sharing, custom user-defined exercises, web app, coaching marketplace, Android/iOS widgets.
