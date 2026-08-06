import React, { useEffect, useRef } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { visibleJoints } from '../../overlay/connections';
import type { CoachState } from '../../coach/pipeline';
import type { ExerciseName } from '../../core/exercises';

interface Props {
  state: CoachState;
  exercise: ExerciseName;
  onPositionOk: () => void;
}

const REQUIRED_CONSECUTIVE_FRAMES = 30;
const VISIBILITY_THRESHOLD = 0.3;

// Exercise-specific positioning guidance
const POSITION_GUIDANCE: Record<ExerciseName, { title: string; body: string; joints: string }> = {
  'Bicep Curl': {
    title: 'SHOW YOUR UPPER BODY',
    body: 'Position your shoulders, elbows, and wrists in the frame',
    joints: 'Shoulders • Elbows • Wrists',
  },
  'Squat': {
    title: 'SHOW YOUR LOWER BODY',
    body: 'Position your hips, knees, and ankles in the frame',
    joints: 'Hips • Knees • Ankles',
  },
  'Lateral Raise': {
    title: 'SHOW YOUR UPPER BODY',
    body: 'Position your shoulders and arms in the frame',
    joints: 'Shoulders • Elbows • Wrists',
  },
  'Shoulder Press': {
    title: 'SHOW YOUR UPPER BODY',
    body: 'Position your shoulders, elbows, and torso in the frame',
    joints: 'Shoulders • Elbows • Torso',
  },
  'Tricep Finisher': {
    title: 'SHOW YOUR UPPER BODY',
    body: 'Position your shoulders, elbows, and wrists in the frame',
    joints: 'Shoulders • Elbows • Wrists',
  },
};

// Key landmark indices that MUST be clearly visible per exercise
// (subset of visibleJoints — only the joints the model actually uses)
const KEY_LANDMARKS: Record<ExerciseName, number[]> = {
  'Bicep Curl': [11, 12, 13, 14, 15, 16],           // shoulders, elbows, wrists
  'Squat': [23, 24, 25, 26, 27, 28],                  // hips, knees, ankles
  'Lateral Raise': [11, 12, 13, 14, 15, 16],          // shoulders, elbows, wrists
  'Shoulder Press': [11, 12, 13, 14, 15, 16, 23, 24], // shoulders, elbows, wrists, hips
  'Tricep Finisher': [11, 12, 13, 14, 15, 16],        // shoulders, elbows, wrists
};

export function PositionCheck({ state, exercise, onPositionOk }: Props) {
  const countRef = useRef(0);
  const guidance = POSITION_GUIDANCE[exercise];
  const keyLandmarks = KEY_LANDMARKS[exercise];

  useEffect(() => {
    const lms = state.smoothedLandmarks;
    if (lms == null) {
      countRef.current = 0;
      return;
    }

    // Check that every key landmark for this exercise is present in the frame
    const allVisible = keyLandmarks.every((idx) => {
      if (lms[idx] == null) return false;
      // Check that the landmark coordinates are within reasonable normalized bounds
      const [x, y] = lms[idx];
      return x >= 0.02 && x <= 0.98 && y >= 0.02 && y <= 0.98;
    });

    if (allVisible) {
      countRef.current += 1;
      if (countRef.current >= REQUIRED_CONSECUTIVE_FRAMES) {
        onPositionOk();
      }
    } else {
      countRef.current = 0;
    }
  }, [state, exercise, onPositionOk, keyLandmarks]);

  const progress = Math.min(countRef.current / REQUIRED_CONSECUTIVE_FRAMES, 1);

  return (
    <View style={styles.overlay} pointerEvents="none">
      <View style={styles.banner}>
        <Text style={styles.title}>{guidance.title}</Text>
        <Text style={styles.sub}>{guidance.body}</Text>
        <View style={styles.jointsRow}>
          <Text style={styles.jointsText}>{guidance.joints}</Text>
        </View>
        {progress > 0 && progress < 1 && (
          <View style={styles.progressContainer}>
            <View style={[styles.progressBar, { width: `${progress * 100}%` }]} />
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFill,
    alignItems: 'center',
    justifyContent: 'center',
  },
  banner: {
    backgroundColor: 'rgba(0, 0, 0, 0.82)',
    paddingVertical: 20,
    paddingHorizontal: 28,
    borderRadius: 20,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#00D2FF',
    maxWidth: '85%',
  },
  title: {
    color: '#00D2FF',
    fontSize: 18,
    fontWeight: '800',
    letterSpacing: 1.5,
  },
  sub: {
    color: '#ffffff',
    fontSize: 14,
    marginTop: 6,
    textAlign: 'center',
    lineHeight: 20,
  },
  jointsRow: {
    marginTop: 10,
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 10,
    backgroundColor: 'rgba(0, 210, 255, 0.12)',
  },
  jointsText: {
    color: '#00D2FF',
    fontSize: 12,
    fontWeight: '600',
    letterSpacing: 0.8,
  },
  progressContainer: {
    width: '100%',
    height: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    borderRadius: 2,
    marginTop: 12,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#39FF14',
    borderRadius: 2,
  },
});
