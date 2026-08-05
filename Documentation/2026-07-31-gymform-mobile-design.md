# GymForm AI Mobile — Technical Design

**Date:** 2026-07-31
**Status:** Approved
**Companion doc:** [2026-07-31-gymform-mobile-prd.md](2026-07-31-gymform-mobile-prd.md) (product requirements)

## 1. System overview

React Native (Expo with dev-client builds — native modules required) app for iOS and Android. The on-device AI form coach is the core; a thin backend supports accounts, sync, social, and AI program generation.

```
┌─────────────────────── Phone ───────────────────────┐
│  React Native app                                   │
│  ├── Live Coach (camera + pose ML, fully on-device) │
│  ├── Workout tracking (local-first, SQLite)         │
│  ├── Programs / streaks / social UI                 │
│  └── Paywall (RevenueCat SDK)                       │
└───────────────┬─────────────────────────────────────┘
                │ sync / auth / social / program generation
        ┌───────▼────────┐      ┌──────────────┐
        │    Supabase    │      │  RevenueCat  │
        │ Postgres, Auth,│      │ (subscription│
        │ Edge Functions,│      │    state)    │
        │ Push           │      └──────────────┘
        └───────┬────────┘
                │ program generation (Edge Function)
        ┌───────▼────────┐
        │  Claude API    │
        │ (claude-opus-5)│
        └────────────────┘
```

**Principles:**
- All CV/ML on-device. No video or landmarks leave the phone — latency requirement and privacy selling point.
- Local-first data: SQLite is the device's source of truth; cloud sync is background/best-effort. Gyms have bad reception.
- Thin backend: Supabase (Postgres + Auth + Edge Functions + push via FCM/APNs). No custom server.
- RevenueCat is the single source of entitlement truth; one cached `isPremium` boolean works offline.

## 2. Live Coach core (port of the existing Python app)

The existing pipeline (Streamlit → MediaPipe → PyTorch MLP → rep state machines) maps to mobile as follows:

| Python component | Mobile replacement |
|---|---|
| Streamlit webcam via WebRTC | `react-native-vision-camera` frame processors |
| MediaPipe Python (`pose_landmarker_lite.task`) | MediaPipe Tasks native Android/iOS SDKs, same `.task` file, LIVE_STREAM mode, GPU delegate — wrapped in a thin owned native module (community plugins as reference, not dependency) |
| PyTorch MLP (8→64→32→10, `gym_model_fullbody.pt`) | One-time export to ONNX, run via `onnxruntime-react-native` (~3k params, sub-ms). Same per-exercise class masking |
| OpenCV skeleton drawing | Skia (`@shopify/react-native-skia`) overlay on the camera view; green/red segments identical to today |
| `exercises.py` state machines, angle math, EMA smoothing, 10-frame majority vote | Ported 1:1 to pure TypeScript modules — unit-testable, no native deps |

**Data flow per frame:** camera frame → frame processor → pose landmarks → EMA smoothing → 8 joint angles → ONNX classifier (masked to selected exercise) → majority-vote smoothing → rep state machine → shared values → Skia overlay + UI stats (UI updates decoupled at ~5/sec; inference runs at camera rate).

**Performance budget:** target 30 fps, floor 15 fps on mid-range devices. Below floor: reduce camera resolution first (480×360 is sufficient — it's what the current app uses), never model quality. Frame processing must never block the UI thread.

**New for mobile:**
- Guided session flow: exercise select → positioning check (all required landmarks visible with adequate confidence) → coached set → rest timer → set summary.
- Voice cues via TTS (`expo-speech`): rep count milestones and correction cues mapped from the existing per-exercise bad-form joint detection (e.g. Bicep Curl shoulder angle > 40° → "keep your elbows in"). Rate-limited to avoid nagging.
- Set summary computes **form quality %** = good reps ÷ attempted reps.

## 3. Data model & sync (Supabase)

**Auth:** Supabase Auth with Apple Sign-In (App Store requirement alongside social login), Google, and email. Anonymous local mode is fully functional; account required only for sync/social. On sign-up, local history merges up.

**Local store:** SQLite (`expo-sqlite`; WatermelonDB if reactive queries prove necessary). Sessions are append-only, single-writer — conflicts effectively nonexistent. Sync engine pushes completed sessions and pulls social data when online.

**Postgres tables:**
- `profiles` — user id, display name, avatar, settings (units, voice on/off, reminder time).
- `workout_sessions` — user, started/ended timestamps, nullable program-day reference.
- `session_sets` — session id, exercise, attempted reps, good reps, form quality %, avg confidence.
- `programs` / `program_days` / `program_exercises` — routine content (authored rows seeded; AI-generated rows created per-user by the Edge Function). A program day = ordered (exercise, sets × reps, rest seconds).
- `user_program_state` — active program, current day, completions.
- `streaks` — current/longest streak, last workout date. Computed by an Edge Function on session upload (server clock, prevents client clock cheating).
- `friendships` — requester, addressee, status.
- `subscriptions` — mirror of RevenueCat entitlements via webhook (server-trustable premium checks for social features).
- Leaderboard = SQL view over accepted friends' weekly good-rep totals.

**Row-Level Security:** users read/write only their own rows; friends' data exposed only through the leaderboard view (aggregates, never raw sessions).

**Push:** Expo Push (FCM/APNs) triggered by scheduled Edge Functions: reminder at user-chosen time, streak warning (evening, streak ≥3), program nudge. Hard cap one/day.

## 4. Monetization

- RevenueCat SDK wraps StoreKit + Play Billing. Monthly and annual (annual has 7-day trial).
- Entitlement check is one cached boolean; premium features work offline.
- Free tier: Bicep Curl + Squat, 7-day history, streaks. Premium: all exercises, unlimited history + charts, programs (authored + AI).
- Paywall triggers: locked exercise tap, program open, history day-8. Never mid-workout.
- RevenueCat webhook → Supabase `subscriptions` mirror.

## 5. AI-tailored programs

**Layer 1 — LLM generation (premium, server-side).** A Supabase Edge Function (Deno/TypeScript) calls the Claude API using the official `@anthropic-ai/sdk`, model `claude-opus-5`, API key in Supabase secrets — never shipped in the app. Input: onboarding quiz answers (goal, experience, days/week, session length, limitations) + summary of the user's recent form-quality stats. Output: a program object validated by **structured outputs** (`output_config.format` with a JSON schema: weeks → days → [exercise ∈ the 5 supported, sets, reps, rest]), inserted into the standard `programs` tables. One call per generation event: quiz completion, goal change, or program-block boundary (~4 weeks) — cost is per-user-per-month, not per-workout.

**Layer 2 — adaptive engine (deterministic, on-device).** Weekly rule-based progression from actual data: all sets completed at ≥85% form quality → +1–2 reps or +1 set; form quality <60% → hold/reduce volume; missed days → redistribute across remaining days. Pure TypeScript, offline, explainable — every adjustment carries a reason string shown to the user. The LLM is re-consulted only at block boundaries, fed the adaptive engine's history summary.

**Failure path:** generation failure → one retry → fall back to the closest authored program (matched on days/week + experience) so onboarding never dead-ends; retry generation silently later.

## 6. Error handling

- **No pose detected:** "step into frame" overlay; coach pauses, doesn't crash.
- **Low confidence (<60%):** "Tracking…" state, no rep penalty (as in the current app).
- **Camera permission denied:** dedicated recovery screen with settings deep-link.
- **Sync failure:** silent exponential-backoff retry; local data authoritative; a subtle "not synced" indicator only.
- **LLM generation failure:** retry once → authored-program fallback (§5).
- **RevenueCat unreachable:** cached entitlement honored.
- **Store/billing errors:** surfaced with retry; never lock a paying user out on transient failure.

## 7. Testing

- **Unit (bulk of coverage):** TS ports of angle math, EMA smoothing, majority vote, all five rep state machines — written test-first against fixture data exported from the Python implementation so ported behavior provably matches; adaptive progression engine rules.
- **Golden-file model tests:** recorded landmark sequences → ONNX model in CI; assert classifications match PyTorch reference outputs.
- **Integration:** Supabase RLS policies (cross-user access must fail), sync engine offline/online transitions, Edge Function schema validation and fallback path.
- **Device matrix:** frame-rate benchmarks on low/mid/high-end devices per release; regression gate at the 15 fps floor.
- **E2E:** Maestro (or Detox) for critical flows — onboarding → first coached set → summary; paywall purchase (sandbox); sign-up → sync merge.

## 8. Build phases

1. **CV core** — camera pipeline, pose module, ONNX classifier, TS logic ports, guided sessions, TTS. No backend. TestFlight/Play-internal beta.
2. **Tracking** — SQLite history, PRs, charts.
3. **Accounts & sync** — Supabase auth, sync engine, streak Edge Function, push.
4. **Monetization + programs** — RevenueCat, paywall, authored programs, AI generation Edge Function, adaptive engine.
5. **Social** — friendships, leaderboard view, share cards. Public launch.

Each phase gets its own implementation plan; Phase 1 de-risks everything downstream and starts first.
