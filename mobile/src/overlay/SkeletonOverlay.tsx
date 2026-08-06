import React from 'react';
import { StyleSheet } from 'react-native';
import Svg, { Circle, Line } from 'react-native-svg';
import { visibleConnections, visibleJoints } from './connections';
import type { CoachState } from '../coach/pipeline';
import type { ExerciseName } from '../core/exercises';

interface Props {
  state: CoachState;
  exercise: ExerciseName;
  width: number;
  height: number;
  mirrorFrontCamera?: boolean;
}

export function SkeletonOverlay({
  state,
  exercise,
  width,
  height,
  mirrorFrontCamera = true,
}: Props) {
  const lms = state.smoothedLandmarks;
  if (lms == null || width === 0 || height === 0) return null;

  const tracking = state.formLabel === 'Tracking...';
  const baseColor = tracking ? '#00FFFF' : '#00FF00';
  const bad = new Set(state.badJoints);

  const getPoint = (idx: number) => {
    const normX = lms[idx][0];
    const normY = lms[idx][1];
    const x = (mirrorFrontCamera ? 1 - normX : normX) * width;
    const y = normY * height;
    return { x, y };
  };

  return (
    <Svg style={styles.svg} width={width} height={height} pointerEvents="none">
      {visibleConnections(exercise).map(([s, e]) => {
        const p1 = getPoint(s);
        const p2 = getPoint(e);
        const isBad = bad.has(s) && bad.has(e);
        return (
          <Line
            key={`conn-${s}-${e}`}
            x1={p1.x}
            y1={p1.y}
            x2={p2.x}
            y2={p2.y}
            stroke={isBad ? '#FF0000' : baseColor}
            strokeWidth={isBad ? 6 : 3}
          />
        );
      })}
      {visibleJoints(exercise).map((i) => {
        const pt = getPoint(i);
        return (
          <Circle
            key={`joint-${i}`}
            cx={pt.x}
            cy={pt.y}
            r={5}
            fill={bad.has(i) ? '#FF0000' : '#FFFFFF'}
          />
        );
      })}
    </Svg>
  );
}

const styles = StyleSheet.create({
  svg: {
    position: 'absolute',
    top: 0,
    left: 0,
  },
});
