import { useCallback, useEffect, useRef, useState } from 'react';
import * as ort from 'onnxruntime-react-native';
import { Asset } from 'expo-asset';
import { CoachPipeline, type CoachState, type PoseResult } from './pipeline';
import { createOnnxClassifier, type OrtLike } from '../ml/onnxClassifier';
import type { ExerciseName } from '../core/exercises';

const IDLE: CoachState = {
  reps: 0,
  stage: 'down',
  formLabel: 'Waiting...',
  confidence: 0,
  smoothedLandmarks: null,
  badJoints: [],
};

export function useCoach(exercise: ExerciseName) {
  const pipelineRef = useRef<CoachPipeline | null>(null);
  const busyRef = useRef(false);
  const [state, setState] = useState<CoachState>(IDLE);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const asset = Asset.fromModule(require('../../assets/models/gym_model_v2.onnx'));
        await asset.downloadAsync();
        const modelUri = asset.localUri || asset.uri;
        const clf = await createOnnxClassifier(modelUri, ort as unknown as OrtLike);
        if (!cancelled) {
          pipelineRef.current = new CoachPipeline(exercise, clf);
        }
      } catch (err) {
        console.error('Failed to load ONNX model asset:', err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    pipelineRef.current?.setExercise(exercise);
    setState(IDLE);
  }, [exercise]);

  const onPose = useCallback((pose: PoseResult | null) => {
    const pipeline = pipelineRef.current;
    if (pipeline == null || busyRef.current) return; // Backpressure: drop frame if busy
    busyRef.current = true;
    pipeline
      .process(pose)
      .then(setState)
      .catch((err) => console.error('Error in coach pipeline:', err))
      .finally(() => {
        busyRef.current = false;
      });
  }, []);

  return { state, onPose };
}
