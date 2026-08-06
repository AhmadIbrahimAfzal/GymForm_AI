import React from 'react';
import { FlatList, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import type { SetResult } from '../session/sessionMachine';
import type { ExerciseName } from '../core/exercises';

interface Props {
  exercise: ExerciseName;
  summary: { setResults: SetResult[]; formQualityPct: number };
  onDone: () => void;
}

export function SetSummaryScreen({ exercise, summary, onDone }: Props) {
  const quality = summary.formQualityPct;
  const toneColor = quality >= 80 ? '#39FF14' : quality >= 60 ? '#FFD700' : '#FF4444';

  return (
    <View style={styles.container}>
      <Text style={styles.header}>WORKOUT SUMMARY</Text>
      <Text style={styles.exerciseTitle}>{exercise}</Text>

      <View style={styles.scoreCard}>
        <Text style={styles.scoreLabel}>FORM QUALITY</Text>
        <Text style={[styles.scoreValue, { color: toneColor }]}>
          {quality.toFixed(0)}%
        </Text>
      </View>

      <Text style={styles.sectionTitle}>Set Breakdown</Text>

      <FlatList
        data={summary.setResults}
        keyExtractor={(_, i) => String(i)}
        renderItem={({ item, index }) => (
          <View style={styles.setRow}>
            <Text style={styles.setText}>Set {index + 1}</Text>
            <Text style={styles.setDetail}>
              {item.good} / {item.attempted} Strict Reps
            </Text>
          </View>
        )}
      />

      <TouchableOpacity style={styles.doneBtn} activeOpacity={0.8} onPress={onDone}>
        <Text style={styles.doneBtnText}>DONE</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
    paddingTop: 60,
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  header: {
    color: 'rgba(255, 255, 255, 0.5)',
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 2,
  },
  exerciseTitle: {
    color: '#00D2FF',
    fontSize: 28,
    fontWeight: 'bold',
    marginVertical: 6,
  },
  scoreCard: {
    backgroundColor: 'rgba(20, 20, 25, 0.9)',
    borderRadius: 20,
    padding: 24,
    alignItems: 'center',
    marginVertical: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  scoreLabel: {
    color: 'rgba(255, 255, 255, 0.5)',
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 1.5,
  },
  scoreValue: {
    fontSize: 48,
    fontWeight: '900',
    marginTop: 4,
  },
  sectionTitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 12,
  },
  setRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    padding: 16,
    borderRadius: 12,
    marginBottom: 10,
  },
  setText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  setDetail: {
    color: '#39FF14',
    fontSize: 16,
    fontWeight: '700',
  },
  doneBtn: {
    backgroundColor: '#00D2FF',
    paddingVertical: 16,
    borderRadius: 16,
    alignItems: 'center',
    marginTop: 20,
  },
  doneBtnText: {
    color: '#000000',
    fontSize: 16,
    fontWeight: '800',
  },
});
