# GymForm AI Mobile — Task Tracker

Tracks progress against the [PRD](docs/superpowers/specs/2026-07-31-gymform-mobile-prd.md) and [technical design](docs/superpowers/specs/2026-07-31-gymform-mobile-design.md). Check items off as they land; add newly discovered tasks under the phase they belong to.

## Phase 0 — Foundations (done)

- [x] Python prototype: Streamlit live coach with MediaPipe + PyTorch (this repo)
- [x] Training pipeline: `build_dataset.py`, `train_model.py`, `dataset_fullbody.csv`
- [x] Trained classifier: `gym_model_fullbody.pt` (10 classes, 5 exercises)
- [x] Rep-counting state machines for all 5 exercises (`exercises.py`)
- [x] Product requirements document (PRD)
- [x] Technical design document
- [x] Phase 1 implementation plan ([plan](docs/superpowers/plans/2026-07-31-phase1-cv-core.md))

## Phase 1 — CV core (live coach)

- [ ] Scaffold React Native (Expo dev-client) project, iOS + Android builds
- [ ] Camera pipeline: `react-native-vision-camera` + frame processor
- [ ] Native MediaPipe pose landmarker module (Android + iOS, `pose_landmarker_lite.task`, GPU delegate)
- [ ] Export PyTorch MLP to ONNX; integrate `onnxruntime-react-native`
- [ ] Port logic to TypeScript with fixture tests against Python reference:
  - [ ] Angle calculation
  - [ ] EMA landmark smoothing
  - [ ] Majority-vote prediction smoothing
  - [ ] Per-exercise class masking
  - [ ] 5 rep state machines
- [ ] Skia skeleton overlay (green/red segments, bad-joint highlighting)
- [ ] Live stats UI (reps / stage / form % ), decoupled from frame rate
- [ ] Guided session flow: exercise select → positioning check → coached set → rest timer → set summary
- [ ] Voice cues via TTS (rate-limited correction cues)
- [ ] Golden-file model tests (landmark fixtures → ONNX vs PyTorch outputs)
- [ ] Device matrix frame-rate benchmark (15 fps floor, resolution fallback)
- [ ] TestFlight / Play internal beta

## Phase 2 — Workout tracking

- [ ] SQLite schema + storage layer (sessions, sets, form quality %)
- [ ] Session history (calendar + per-exercise views)
- [ ] Personal records
- [ ] Progress charts (volume, form-quality trend)

## Phase 3 — Accounts & sync

- [ ] Supabase project: schema, RLS policies, migrations
- [ ] Auth: Apple / Google / email; anonymous mode preserved
- [ ] Local→cloud merge on sign-up
- [ ] Background sync engine (append-only push, pull social data)
- [ ] Streak computation Edge Function
- [ ] Push notifications (reminder, streak warning, program nudge; 1/day cap)
- [ ] RLS integration tests (cross-user access must fail)

## Phase 4 — Monetization + programs

- [ ] RevenueCat integration (monthly + annual w/ 7-day trial)
- [ ] Paywall UI + trigger points (locked exercise, program open, history day-8)
- [ ] RevenueCat webhook → `subscriptions` mirror table
- [ ] Authored program content + seeding
- [ ] Guided program-day flow in the live coach
- [ ] Onboarding quiz
- [ ] AI program generation Edge Function (Claude `claude-opus-5`, structured outputs, authored-program fallback)
- [ ] Adaptive progression engine (deterministic, on-device, explainable adjustments)

## Phase 5 — Social + launch

- [ ] Friendships (username / invite link)
- [ ] Weekly good-reps leaderboard (friends-only view)
- [ ] Shareable session-summary card (image only)
- [ ] Store listings, screenshots, privacy labels
- [ ] Public launch
