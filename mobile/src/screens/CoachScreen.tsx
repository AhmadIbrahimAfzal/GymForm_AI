import React, { useEffect, useRef, useState } from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import * as Speech from 'expo-speech';
import {
  Camera,
  useCameraDevice,
  useCameraFormat,
  useCameraPermission,
  useFrameProcessor,
  VisionCameraProxy,
} from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';
import { useCoach } from '../coach/useCoach';
import { SkeletonOverlay } from '../overlay/SkeletonOverlay';
import { StatCard } from './components/StatCard';
import { PositionCheck } from './components/PositionCheck';
import { RestTimer } from './components/RestTimer';
import { SessionMachine } from '../session/sessionMachine';
import { VoiceCoach } from '../session/voiceCoach';
import { FpsMeter, pickResolution } from '../coach/fpsMeter';
import type { ExerciseName } from '../core/exercises';
import type { PoseResult } from '../coach/pipeline';

interface Props {
  exercise: ExerciseName;
  onFinishSession: (summary: ReturnType<SessionMachine['summary']>) => void;
  onCancel: () => void;
}

const posePlugin = VisionCameraProxy.initFrameProcessorPlugin('detectPose', {});

export function CoachScreen({ exercise, onFinishSession, onCancel }: Props) {
  const [layout, setLayout] = useState({ width: 0, height: 0 });
  const [isMuted, setIsMuted] = useState(false);
  const [measuredFps, setMeasuredFps] = useState(30);

  const { hasPermission, requestPermission } = useCameraPermission();
  const { state, onPose } = useCoach(exercise);

  // Session machine & Voice coach instances
  const sessionRef = useRef(
    new SessionMachine({ exercise, targetReps: 10, totalSets: 3, restSeconds: 45 }),
  );
  const voiceRef = useRef(new VoiceCoach((t) => Speech.speak(t), 4000));
  const fpsMeterRef = useRef(new FpsMeter());
  const prevRepsRef = useRef(0);
  const [, forceUpdate] = useState({});

  const session = sessionRef.current;
  const device = useCameraDevice('front');
  const resolution = pickResolution(measuredFps);

  const format = useCameraFormat(device, [
    { videoResolution: resolution },
    { fps: 30 },
  ]);

  // Handle reps increase & form attempt tracking
  useEffect(() => {
    if (session.phase === 'active') {
      if (state.reps > prevRepsRef.current) {
        session.recordAttempt(true);
        session.onRepCounted(state.reps);
        prevRepsRef.current = state.reps;
        forceUpdate({});
      }
    }
  }, [state.reps, session]);

  // Voice cues
  useEffect(() => {
    if (!isMuted) {
      voiceRef.current.onState(state, exercise, Date.now());
    }
  }, [state, exercise, isMuted]);

  const onPoseJS = Worklets.createRunOnJS((pose: PoseResult | null) => {
    fpsMeterRef.current.tick(Date.now());
    const currentFps = fpsMeterRef.current.fps();
    if (currentFps > 0) setMeasuredFps(currentFps);
    onPose(pose);
  });

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      if (posePlugin == null) return;
      const result = posePlugin.call(frame) as unknown as PoseResult | null;
      onPoseJS(result);
    },
    [onPoseJS],
  );

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.title}>GymForm AI</Text>
        <Text style={styles.msg}>Camera access is needed to coach your form.</Text>
      </View>
    );
  }

  if (device == null) {
    return (
      <View style={styles.center}>
        <Text style={styles.msg}>No camera device found.</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Top Navigation / Controls Bar */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.headerBtn} onPress={onCancel}>
          <Text style={styles.headerBtnText}>✕ EXIT</Text>
        </TouchableOpacity>
        <Text style={styles.exerciseName}>{exercise}</Text>
        <TouchableOpacity
          style={styles.headerBtn}
          onPress={() => setIsMuted((m) => !m)}
        >
          <Text style={styles.headerBtnText}>{isMuted ? '🔇' : '🔊'}</Text>
        </TouchableOpacity>
      </View>

      {/* Camera View & Skia Overlay */}
      <View
        style={styles.cameraContainer}
        onLayout={(e) =>
          setLayout({
            width: e.nativeEvent.layout.width,
            height: e.nativeEvent.layout.height,
          })
        }
      >
        <Camera
          style={StyleSheet.absoluteFill}
          device={device}
          format={format}
          isActive={true}
          frameProcessor={frameProcessor}
        />

        <SkeletonOverlay
          state={state}
          exercise={exercise}
          width={layout.width}
          height={layout.height}
          mirrorFrontCamera={true}
        />

        {/* Phase Overlays */}
        {session.phase === 'positioning' && (
          <PositionCheck
            state={state}
            exercise={exercise}
            onPositionOk={() => {
              session.onPositionOk();
              prevRepsRef.current = 0;
              forceUpdate({});
            }}
          />
        )}

        {session.phase === 'rest' && (
          <RestTimer
            restSeconds={session.opts.restSeconds}
            onRestFinished={() => {
              session.onRestFinished();
              forceUpdate({});
            }}
          />
        )}
      </View>

      {/* Live Stats Row */}
      <View style={styles.statsRow}>
        <StatCard label="Reps" value={String(state.reps)} />
        <StatCard label="Stage" value={state.stage.toUpperCase()} />
        <StatCard
          label="Form"
          value={`${state.formLabel} (${state.confidence.toFixed(0)}%)`}
          tone={
            state.formLabel.includes('Good')
              ? 'good'
              : state.formLabel.includes('Bad')
              ? 'bad'
              : 'neutral'
          }
        />
      </View>

      {/* Summary Trigger */}
      {session.phase === 'summary' && (
        <View style={styles.summaryBanner}>
          <TouchableOpacity
            style={styles.summaryBtn}
            onPress={() => onFinishSession(session.summary())}
          >
            <Text style={styles.summaryBtnText}>VIEW WORKOUT SUMMARY</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: 50,
    paddingBottom: 12,
    paddingHorizontal: 16,
    backgroundColor: '#0a0a0a',
    zIndex: 10,
  },
  headerBtn: {
    padding: 8,
  },
  headerBtnText: {
    color: '#00D2FF',
    fontSize: 14,
    fontWeight: '700',
  },
  exerciseName: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  cameraContainer: {
    flex: 1,
    overflow: 'hidden',
  },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#0a0a0a',
  },
  title: {
    color: '#00D2FF',
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 12,
  },
  msg: {
    color: '#ffffff',
    fontSize: 16,
  },
  statsRow: {
    flexDirection: 'row',
    padding: 8,
    backgroundColor: '#0a0a0a',
  },
  summaryBanner: {
    padding: 16,
    backgroundColor: '#0a0a0a',
  },
  summaryBtn: {
    backgroundColor: '#39FF14',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  summaryBtnText: {
    color: '#000000',
    fontSize: 16,
    fontWeight: '800',
  },
});
