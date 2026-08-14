# GymForm AI — Implementation & Session-Wise Development Log

> [!NOTE]
> This document tracks the implementation plan, roadmap, architectural decisions, and session-by-session progress for **GymForm AI**. It is maintained and updated after every development session.

---

## 📌 Project Overview & Architecture

GymForm AI is a real-time, privacy-first mobile fitness coach powered by on-device computer vision.
- **Frontend**: React Native (Expo SDK 57), TypeScript, VisionCamera v5, SVG Skeleton Overlay.
- **Native ML Pipeline**: Kotlin FrameProcessor Plugin calling MediaPipe Pose Landmarker (`pose_landmarker_lite.task`), ONNX Runtime (`onnxruntime-react-native`).
- **Classifier**: Upgraded PyTorch v2 MLP (`gym_model_v2.onnx`), 8 joint angle inputs, StandardScaler normalized, 10-class form classification head.
- **Session Engine**: Core state machines tracking exercise reps, exercise-specific joint visibility, form quality percentage, and voice coaching cues via TTS.

---

## 📜 Session-Wise Development Log

### Session 1: AI Model & Dataset Upgrade (PyTorch → ONNX)
- **Data Augmentation**: Created `augment_dataset.py`, expanded 5,401 raw samples → 76,960 balanced rows (mirroring, noise injection, class balancing).
- **v2 Model Training**: Developed `train_model_v2.py` with BatchNorm, LeakyReLU, and 128→64→32 architecture. Trained `gym_model_v2.pt` (72% test accuracy on augmented data).
- **Scaler Export**: Saved `scaler_params.json` (means & standard deviations for 8 joint angles) to enable feature normalization.
- **ONNX Export**: Created `tools/export_onnx_v2.py`, exported `gym_model_v2.onnx` (2.6 KB), verified PyTorch vs ONNX output parity ($<3 \times 10^{-5}$ diff).
- **Assets Bundled**: Placed ONNX model and scaler params into `mobile/assets/models/`.

### Session 2: Core TypeScript Engine & ONNX Classifier Integration
- **Scaffold**: Configured `mobile/` directory with strict TypeScript, Expo SDK 57, and Metro asset extensions (`.onnx`, `.task`).
- **Angle Math & Core Filters**: Implemented `calculateAngle()`, `computeJointAngles()` ([angles.ts](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/src/core/angles.ts)), EMA filter ([ema.ts](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/src/core/ema.ts)), and 10-frame Majority Vote ([majorityVote.ts](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/src/core/majorityVote.ts)).
- **Classifier & Scaler**: Implemented `anglesToInput()` and `classify()` in [classifier.ts](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/src/core/classifier.ts) with `StandardScaler` feature normalization `(x - mean) / std`.
- **Exercise State Machines**: Built 5 exercise state machines ([bicepCurl.ts](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/src/core/exercises/bicepCurl.ts), [squat.ts](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/src/core/exercises/squat.ts), [lateralRaise.ts](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/src/core/exercises/lateralRaise.ts), [shoulderPress.ts](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/src/core/exercises/shoulderPress.ts), [tricepFinisher.ts](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/src/core/exercises/tricepFinisher.ts)).
- **ONNX Mobile Wrapper**: Built [onnxClassifier.ts](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/src/ml/onnxClassifier.ts) using `onnxruntime-react-native`.

### Session 3: Native Android Kotlin FrameProcessor Plugin
- **MediaPipe Integration**: Added `com.google.mediapipe:tasks-vision:0.10.14` to Android build dependencies. Placed `pose_landmarker_lite.task` in Android assets.
- **Native Plugin**: Created [PoseFrameProcessorPlugin.kt](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/android/app/src/main/java/com/gymform/mobile/posedetector/PoseFrameProcessorPlugin.kt) and [PoseFrameProcessorPluginPackage.kt](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/android/app/src/main/java/com/gymform/mobile/posedetector/PoseFrameProcessorPluginPackage.kt).
- **Application Registration**: Registered plugin package and `OnnxruntimePackage` in [MainApplication.kt](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/android/app/src/main/java/com/gymform/mobile/MainApplication.kt).

### Session 4: UI, Session Flow, Skeleton Overlay & Voice Coach
- **Session Machine**: Built [sessionMachine.ts](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/src/session/sessionMachine.ts) (positioning → active → rest timer → summary).
- **Voice Coach**: Implemented [voiceCoach.ts](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/src/session/voiceCoach.ts) using `expo-speech` with 4-second cue cooldown.
- **Skeleton Overlay**: Created [SkeletonOverlay.tsx](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/src/overlay/SkeletonOverlay.tsx) rendering green skeleton lines for good form and red for incorrect joints.
- **UI Screens**: Developed [ExerciseSelectScreen.tsx](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/src/screens/ExerciseSelectScreen.tsx), [CoachScreen.tsx](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/src/screens/CoachScreen.tsx), and [SetSummaryScreen.tsx](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/src/screens/SetSummaryScreen.tsx).

### Session 5: Native Android Crash Fix, Workout-Specific Positioning & Build Diagnostics
- **Native Crash Fix**: Resolved `ImageAnalysis` frame processor crash by converting VisionCamera YUV frames to an ARGB `Bitmap` copy before passing to MediaPipe (`BitmapImageBuilder`), preventing recycled image exceptions.
- **ImageProxy Dependency Fix**: Removed explicit `CameraX ImageProxy` references that caused Gradle compilation failures, switching to VisionCamera's native `frame.orientation.unionValue`.
- **Workout-Specific PositionCheck**: Refactored [PositionCheck.tsx](file:///c:/Users/ahmad/OneDrive/Desktop/Gym_AI_Project%20App/mobile/src/screens/components/PositionCheck.tsx) to provide exercise-specific instructions (e.g. "SHOW YOUR UPPER BODY" for curls vs "SHOW YOUR LOWER BODY" for squats). Replaced full-body requirements with relevant landmark checks and added a progress bar.
- **Style Fixes**: Replaced invalid React Native `justify` properties with `justifyContent` across `CoachScreen`, `StatCard`, and `RestTimer`. Replaced deprecated `StyleSheet.absoluteFillObject` with `StyleSheet.absoluteFill`.
- **Type Check**: Verified 100% clean TypeScript build (`npx tsc --noEmit`).

---

## 🗺️ Master Roadmap & Phase Checklist

### Phase 0 — Foundation (DONE)
- [x] Streamlit Python prototype with PyTorch + MediaPipe
- [x] Initial dataset (5,401 frames) & 5 exercise state machines
- [x] PRD & Architecture Specification

### Phase 1 — AI Model Enhancement (IN PROGRESS)
- [x] Data augmentation pipeline (`augment_dataset.py`)
- [x] v2 PyTorch model training with StandardScaler normalization & BatchNorm
- [x] ONNX export & parity validation
- [ ] **1A. Diverse Body Data Collection**: Record 5+ individuals with varied heights/body types for improved generalizability
- [ ] **1B. New Exercise Support**: Collect data for Deadlift, Pushup, Plank (expand output head to 16 classes)
- [ ] **1C. Retraining & Re-export**: Re-run augmentation, train v3 model, export updated ONNX asset

### Phase 2 — Core CV Mobile App (CURRENT PHASE)
- [x] 2A. Core TypeScript Logic & Angle Math
- [x] 2B. ONNX Model Integration & Normalization Pipeline
- [x] 2C. Camera & Native Kotlin MediaPipe Frame Processor
- [x] 2D. End-to-End Pipeline Wiring (`CoachPipeline`)
- [x] 2E. Live Coaching UI, Skeleton Overlay & Workout-Specific Positioning
- [x] 2F. Voice Cues & FPS Fallbacks
- [ ] **2G. On-Device APK Verification**: Verify ADB deployment & live camera coaching on connected Android Pixel device

### Phase 3 — Local Workout History & Analytics (PLANNED)
- [ ] SQLite local storage schema (`expo-sqlite`)
- [ ] Session history screen & personal record (PR) tracking
- [ ] Form quality trend charts over time

### Phase 4 — Accounts & Cloud Sync (PLANNED)
- [ ] Supabase backend setup (Postgres, RLS policies)
- [ ] Social auth (Google/Apple) & offline-first sync engine
- [ ] Server-side workout streak calculations & push notifications

### Phase 5 — Monetization & Programs (PLANNED)
- [ ] RevenueCat integration for subscription management
- [ ] Freemium tiering (Curl & Squat free; full library premium)
- [ ] AI-generated multi-week workout programs

---

## 📝 Document Update Protocol

> [!IMPORTANT]
> At the end of every development session:
> 1. Append a new entry under **Session-Wise Development Log** detailing completed fixes, features, and refactorings.
> 2. Update task checkboxes in the **Master Roadmap & Phase Checklist**.
> 3. Save updates to both `implementation_plan.md` in the workspace root and the active artifact.
