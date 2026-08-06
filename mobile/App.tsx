import React, { useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { ExerciseSelectScreen } from './src/screens/ExerciseSelectScreen';
import { CoachScreen } from './src/screens/CoachScreen';
import { SetSummaryScreen } from './src/screens/SetSummaryScreen';
import type { ExerciseName } from './src/core/exercises';
import type { SetResult } from './src/session/sessionMachine';

type Screen = 'select' | 'coaching' | 'summary';

export default function App() {
  const [screen, setScreen] = useState<Screen>('select');
  const [selectedExercise, setSelectedExercise] = useState<ExerciseName>('Bicep Curl');
  const [lastSummary, setLastSummary] = useState<{
    setResults: SetResult[];
    formQualityPct: number;
  }>({ setResults: [], formQualityPct: 0 });

  return (
    <>
      <StatusBar style="light" />

      {screen === 'select' && (
        <ExerciseSelectScreen
          onSelectExercise={(ex) => {
            setSelectedExercise(ex);
            setScreen('coaching');
          }}
        />
      )}

      {screen === 'coaching' && (
        <CoachScreen
          exercise={selectedExercise}
          onFinishSession={(summary) => {
            setLastSummary(summary);
            setScreen('summary');
          }}
          onCancel={() => setScreen('select')}
        />
      )}

      {screen === 'summary' && (
        <SetSummaryScreen
          exercise={selectedExercise}
          summary={lastSummary}
          onDone={() => setScreen('select')}
        />
      )}
    </>
  );
}
