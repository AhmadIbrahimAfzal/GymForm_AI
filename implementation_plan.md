# Analysis: Your New Documents vs Our Roadmap

I've read all 4 documents cover to cover. Here's the deep breakdown.

---

## What Each Document Is

| Document | What It Is | Size | Quality |
|----------|-----------|------|---------|
| [PRD](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/2026-07-31-gymform-mobile-prd.md) | Product Requirements Document — defines *what* the app does, who it's for, monetization, feature scope, release phases | 120 lines | **Excellent** — professional-grade, well-structured, clear feature tiers |
| [Design](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/2026-07-31-gymform-mobile-design.md) | Technical Design — defines *how* the app is built, architecture, tech stack, data model, backend | 124 lines | **Very good** — clear system diagram, smart technology choices |
| [Phase 1 Core](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/2026-07-31-phase1-cv-core.md) | Implementation plan for Phase 1 — the live coach MVP. 19 tasks, step-by-step, with exact code, tests, and commit messages | 2,201 lines | **Outstanding** — the most detailed implementation plan I've ever seen. Every line of code is specified |
| [TASKS.md](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/TASKS.md) | Master task tracker across all 6 phases | 70 lines | **Good** — clean checklist, properly phased |

---

## Viability Assessment: Can This Ship?

### The short answer: **Yes, Phase 1 is absolutely viable and will produce a working Play Store beta.**

### The longer answer:

The Phase 1 plan (CV Core) is the most important document. It takes you from zero to a **working AI form coach app** with:
- Camera → pose detection → form classification → rep counting → skeleton overlay → voice cues
- Guided session flow (exercise select → positioning check → coached set → rest timer → summary)
- Internal beta via EAS Build (Play Store internal track)

That's a real, shippable MVP. The plan is broken into 19 granular tasks with test-driven development, fixture parity testing against your Python code, and proper native module architecture. **This is production-level planning.**

---

## Pros (What These Documents Get Right)

### PRD

| Pro | Why It Matters |
|-----|---------------|
| **"Form quality %" as the signature metric** | This is genuinely novel — no fitness app tracks good reps vs attempted reps. It's a real differentiator |
| **Privacy-first (no video leaves device)** | Smart positioning for App Store and user trust. A real selling point |
| **Freemium with clear tier boundaries** | Free = Curl + Squat (prove value). Premium = all exercises + programs + history. Clean paywall logic |
| **Phased release (5 phases)** | Doesn't try to build everything at once. Phase 1 (CV core) de-risks everything |
| **Realistic target users** | Gym beginners 18-35 is the right audience. They're the most likely to pay for form help |
| **Out-of-scope list** | Explicitly saying NO to wearables, nutrition, web app, etc. is mature product thinking |

### Design

| Pro | Why It Matters |
|-----|---------------|
| **Hybrid on-device + thin backend** | All ML on device (privacy + latency), Supabase only for auth/sync/social. Clean separation |
| **SQLite local-first** | Offline-first is crucial for gyms. Data syncs when possible, never blocks |
| **ONNX runtime for the classifier** | Smart choice — your tiny MLP runs in sub-millisecond via ONNX. No need for TFLite conversion |
| **RevenueCat for subscriptions** | Industry standard. Handles Apple/Google billing, trials, entitlements. One cached boolean for offline |
| **Claude API for program generation** | Using an LLM constrained to supported exercises via structured outputs is clever — avoids hallucinating unsupported exercises |
| **Supabase Edge Functions** | Serverless, free tier is generous, no managing infrastructure |

### Phase 1 Core Plan

| Pro | Why It Matters |
|-----|---------------|
| **Test-first with Python parity fixtures** | The `tools/export_fixtures.py` approach is brilliant — generates ground truth from your Python code, then the TS tests must match exactly. This guarantees the port is correct |
| **Clean architecture** | `mobile/src/core/` is pure TypeScript with zero React/native dependencies. Unit-testable in Node. This is textbook separation of concerns |
| **Frame-processor plugin architecture** | Writing your own thin native wrapper around MediaPipe (rather than depending on a community RN package) is the right call. Community packages break constantly |
| **Backpressure handling** | The `busyRef` pattern in `useCoach` — if inference is in flight, drop the frame. This prevents frame accumulation and memory issues |
| **FPS meter + resolution fallback** | Automatically downgrading from 480×360 to 320×240 when fps < 15 is a great UX safety net |
| **Voice cues with cooldown** | Rate-limited to 4s between cues. Prevents nagging while still being useful |
| **Conventional commits** | Every task has a commit message. Good for CI and changelog generation |

### TASKS.md

| Pro | Why It Matters |
|-----|---------------|
| **Phase 0 marked as done** | Correctly credits the existing Python prototype as completed foundation work |
| **Phase 1 breakdown matches the core plan** | 1:1 mapping, no disconnect between documents |
| **Clear phase dependencies** | Each phase builds on the previous one |

---

## Cons (What These Documents Get Wrong or Miss)

### PRD — Cons

| Issue | Severity | Detail |
|-------|----------|--------|
| **Scope creep risk in the PRD itself** | ⚠️ Medium | The PRD defines 5 phases spanning AI programs, social, leaderboards, adaptive engines. This is 12-18 months of work for a small team. The PRD is aspirational, not scoped to what you can actually build |
| **"50K installs in 12 months"** | ⚠️ Medium | Ambitious for an indie app with no marketing budget. A more realistic target is 5-10K installs |
| **Revenue model assumes 3% conversion** | ⚠️ Medium | Industry average for fitness apps is 1-2%. 3% requires an exceptional product and strong retention |
| **No mention of model improvement** | 🔴 High | The PRD treats the AI model as a given. It never mentions improving the classifier. This is a critical gap (see below) |
| **"AI-tailored programs" via Claude** | ⚠️ Medium | This is expensive (Claude API calls per user per month) and complex to validate. Should be Phase 5, not Phase 4 |

### Design — Cons

| Issue | Severity | Detail |
|-------|----------|--------|
| **Uses the OLD model (`gym_model_fullbody.pt`)** | 🔴 High | The design exports the original 8→64→32→10 model, not our improved v2 (8→128→64→32→10 with BatchNorm). The v2 model we trained is sitting right there as `gym_model_v2.pt` |
| **No feature normalization in the ONNX pipeline** | 🔴 High | The ONNX export script in Task 8 loads the model and exports raw. But our v2 model was trained with StandardScaler normalization. If you use v2, you MUST normalize the 8 angles before inference using the `scaler_params.json` we saved. The design doesn't mention this at all |
| **Claude `claude-opus-5` model reference** | ⚠️ Low | Minor: model name may not exist yet. Use whatever the current best Claude model is at the time |
| **No model versioning strategy** | ⚠️ Medium | How do you update the ONNX model in the app? Currently it's bundled as an asset — updating requires an app update. Should plan for OTA model delivery |
| **Supabase free tier limits not discussed** | ⚠️ Medium | Supabase free tier: 50K monthly active users, 500MB database, 1GB file storage. Fine for MVP but worth knowing |

### Phase 1 Core Plan — Cons

| Issue | Severity | Detail |
|-------|----------|--------|
| **Exports the WRONG model** | 🔴 Critical | Task 8 (`export_onnx.py`) exports `gym_model_fullbody.pt` (the old model). It should export `gym_model_v2.pt` with the v2 architecture. The GymModel class in the export script is the old architecture |
| **No normalization in inference** | 🔴 Critical | `anglesToInput()` in Task 7 sends raw angles. If using v2, it must normalize using scaler_params.json's mean/std first. This is missing from the entire pipeline |
| **Mac-specific paths** | ⚠️ Low | Task 1 uses `/Users/admin/Projects/ReactNative/GymForm_AI` — you're on Windows. Path needs adjusting |
| **iOS tasks may not apply** | ⚠️ Medium | Task 12 (iOS pose plugin) requires a Mac + Apple Developer account ($99/year). If you're Android-only for MVP, skip this |
| **No UI design/styling details** | ⚠️ Medium | The plan specifies behavior but not aesthetics. The exercise select screen, coach screen layout, and summary screen have no design specs. You'll need to design these |
| **No error recovery for ONNX load failure** | ⚠️ Low | If the ONNX model fails to load, the app silently has no pipeline. Should show an error state |
| **`expo prebuild` warning** | ⚠️ Medium | The native pose plugin files live in `android/` which `expo prebuild --clean` regenerates. The plan acknowledges this but doesn't solve it — should be an Expo config plugin |

### TASKS.md — Cons

| Issue | Severity | Detail |
|-------|----------|--------|
| **No model improvement phase** | 🔴 High | There's no phase or task for improving the AI model. The tasks go straight from "trained classifier" (Phase 0) to "build the app" (Phase 1). Our entire Phase 1 (model improvement) is missing |
| **No timeline estimates** | ⚠️ Medium | No time estimates per phase. Hard to plan without knowing each phase takes 2 weeks vs 2 months |
| **No team assignment** | ⚠️ Low | If there are multiple people, who does what? |

---

## Comparison with Our Roadmap

| Topic | Our Roadmap | These New Documents | Verdict |
|-------|------------|--------------------|---------| 
| **Model improvement** | Entire Phase 1 dedicated to it (normalization, augmentation, architecture upgrade, data expansion) | **Completely absent.** Uses the old model as-is | 🔴 Our roadmap is right. The model MUST be improved. 72% accuracy on augmented data (and likely lower on real-world users) is not good enough for a paid product |
| **Data augmentation** | Created `augment_dataset.py`, expanded from 5,401 → 76,960 balanced rows | Not mentioned | Our work is already done and usable |
| **Training pipeline** | Created `train_model_v2.py` with proper ML practices | Not mentioned | Our work is already done. The v2 model exists |
| **Model export** | Planned PyTorch → ONNX → TFLite chain | Uses ONNX only (fine for React Native) | Both valid, ONNX-only is cleaner for RN |
| **App framework** | Compared Flutter vs RN vs Kotlin | Chose React Native (Expo) | Aligned. Good choice since you know TS |
| **Inference location** | Compared cloud vs on-device vs hybrid | On-device only (ONNX) | Aligned, though the hybrid approach is worth revisiting later |
| **Architecture** | High-level system diagram | Extremely detailed — per-file, per-function, per-test | New docs are far more detailed |
| **Session flow** | Not specified in our roadmap | Fully designed (positioning check, rest timer, summary, form quality %) | New docs add significant value here |
| **Voice cues** | Mentioned as a feature | Fully designed with cue mapping, cooldown, tests | New docs add significant value |
| **Testing strategy** | Not specified | Comprehensive — fixture parity, golden files, unit tests, device matrix | New docs are much stronger here |
| **Monetization** | Not discussed | Full freemium model with RevenueCat | New docs add this entirely |
| **Backend** | Not discussed (was for later) | Supabase architecture for Phase 3+ | New docs plan ahead well |

---

## What's Missing for the App to Go Live

### Critical (Must Fix Before Beta)

| # | Gap | Impact | Fix |
|---|-----|--------|-----|
| 1 | **Use the improved model (v2)** | Old model has worse accuracy, no normalization, no class balancing | Update Task 8's export script to load `gym_model_v2.pt` with the `GymModelV2` architecture. Add normalization using `scaler_params.json` to `anglesToInput()` |
| 2 | **Add scaler normalization to the inference pipeline** | Without it, v2 model receives un-normalized inputs and produces garbage predictions | In `classifier.ts`, load `scaler_params.json` and normalize: `(angle - mean) / std` for each of the 8 features before sending to ONNX |
| 3 | **UI/UX design** | The plan specifies behavior but not how screens look. A form-coaching app needs a polished, confident UI | Design the exercise select, coach, summary, and settings screens. Use the existing Streamlit dark theme as a starting point |
| 4 | **App icon + splash screen** | Required for Play Store submission | Design and configure in `app.json` |
| 5 | **Privacy policy** | Required by Play Store for camera-using apps | Write one. Key points: no video leaves device, only aggregate workout stats sync |
| 6 | **Play Store listing** | Screenshots, description, feature graphic | Create after the app is functional |

### Important (Should Fix Before Public Launch)

| # | Gap | Impact | Fix |
|---|-----|--------|-----|
| 7 | **More training data from diverse bodies** | Model is trained on your body types only — will fail on different proportions | Record 5+ different people doing each exercise. Re-run augment + train pipeline |
| 8 | **Handling of new exercises** | PRD wants 8-10 exercises but model only supports 5 | Record data for Deadlift, Pushup, Plank. Retrain with new classes. Update model output head |
| 9 | **Crash/error analytics** | You won't know if the app crashes for users | Add Sentry or Firebase Crashlytics |
| 10 | **Onboarding flow** | New users don't know how to position themselves | Add a brief tutorial/demo before first workout |

---

## What's Missing for the AI to Get Way Better

This is the biggest gap in the new documents. **They don't address model improvement at all.** Here's what needs to happen:

### Already Done (From Our Work)

| Improvement | Status | Impact |
|-------------|--------|--------|
| Feature normalization (StandardScaler) | ✅ Done in `train_model_v2.py` | Immediate accuracy boost |
| Data augmentation (mirror + noise + balance) | ✅ Done in `augment_dataset.py` — 5,401 → 76,960 rows | Prevents overfitting to your body types |
| Class balancing | ✅ Done — all 10 classes have 7,696 samples | Model no longer biased toward over-represented classes |
| Better architecture (BatchNorm + LeakyReLU + wider) | ✅ Done in v2 model | Better gradient flow, more capacity |
| Proper train/val/test split + early stopping | ✅ Done | Prevents overfitting, gives reliable accuracy numbers |

### Still Needed (Not in Any Document)

| Improvement | Effort | Expected Impact |
|-------------|--------|----------------|
| **Diverse training data** (5+ different people, multiple angles, lighting) | High (need volunteers) | 🔴 **This is the #1 thing that will make the AI better.** Your model is trained on 2-3 people. It will struggle with different body shapes |
| **Temporal model** (1D CNN + GRU on frame sequences instead of single frames) | Medium (architecture change) | High — can detect motion patterns like "swinging" that single-frame classification misses |
| **Angular velocity + acceleration features** | Low (add 2 more features per joint) | Medium — jerky movements are a form problem that static angles can't detect |
| **Confusion matrix analysis → targeted retraining** | Low | Medium — from our training: Bad Curl (45% recall), Bad Shoulder (46% recall) are the weakest. Need more varied "bad" examples for these |
| **User feedback loop** | Medium | High — let users flag "this was wrong" → collect anonymized angle data → retrain periodically |
| **Upgrade MediaPipe model** | Low (swap `.task` file) | Low-Medium — use `pose_landmarker_full.task` or `heavy.task` for training data extraction (keep lite for real-time) |
| **Per-exercise specialized models** | Medium | Medium — instead of one 10-class model, train 5 binary classifiers (good/bad per exercise). Each can be more accurate for its exercise |
| **Cross-validation instead of single split** | Low | Low — gives more robust accuracy estimates |

---

## Final Verdict

### Are these documents viable and good for deployment?

**Yes.** The PRD + Design + Phase 1 Core Plan form one of the most thorough mobile app blueprints I've seen. If you execute the Phase 1 plan, you will have a working app on the Play Store's internal testing track.

### What's the #1 gap?

**The AI model itself.** The documents treat the model as "done" (Phase 0 ✅). But the model is the core product — and at 72% accuracy (on augmented data from a few people), it's not good enough for real users. You need to:

1. **Use the v2 model we already trained** (not the old one the plan references)
2. **Add normalization to the mobile inference pipeline** (or the v2 model is useless)
3. **Record diverse training data** (this is the single biggest improvement possible)
4. **Eventually upgrade to a temporal model** (but this can be a v2 update)

### Recommended action

Merge both plans:
- Use the new documents' **app architecture + session flow + testing strategy** (it's better than our app plan)
- Use our roadmap's **model improvement work** (it's completely absent from theirs)
- Fix the 2 critical bugs (wrong model export, missing normalization) before writing any app code
