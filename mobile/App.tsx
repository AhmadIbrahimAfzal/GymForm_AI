import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { CoachScreen } from './src/screens/CoachScreen';

export default function App() {
  return (
    <>
      <StatusBar style="light" />
      <CoachScreen />
    </>
  );
}
