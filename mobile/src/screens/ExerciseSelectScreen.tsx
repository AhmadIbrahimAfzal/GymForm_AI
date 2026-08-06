import React from 'react';
import { FlatList, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { EXERCISES, type ExerciseName } from '../core/exercises';

interface Props {
  onSelectExercise: (exercise: ExerciseName) => void;
}

const EXERCISE_LIST = Object.keys(EXERCISES) as ExerciseName[];

export function ExerciseSelectScreen({ onSelectExercise }: Props) {
  return (
    <View style={styles.container}>
      <Text style={styles.header}>GymForm AI</Text>
      <Text style={styles.subHeader}>Select an exercise to begin coaching</Text>

      <FlatList
        data={EXERCISE_LIST}
        keyExtractor={(item) => item}
        contentContainerStyle={styles.list}
        renderItem={({ item }) => (
          <TouchableOpacity
            style={styles.card}
            activeOpacity={0.8}
            onPress={() => onSelectExercise(item)}
          >
            <Text style={styles.cardTitle}>{item}</Text>
            <Text style={styles.cardSub}>3 Sets • 10 Strict Reps</Text>
          </TouchableOpacity>
        )}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
    paddingTop: 60,
    paddingHorizontal: 20,
  },
  header: {
    color: '#00D2FF',
    fontSize: 32,
    fontWeight: 'bold',
  },
  subHeader: {
    color: 'rgba(255, 255, 255, 0.6)',
    fontSize: 16,
    marginTop: 6,
    marginBottom: 24,
  },
  list: {
    paddingBottom: 40,
  },
  card: {
    backgroundColor: 'rgba(20, 20, 25, 0.9)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(0, 210, 255, 0.25)',
  },
  cardTitle: {
    color: '#ffffff',
    fontSize: 20,
    fontWeight: '700',
  },
  cardSub: {
    color: '#39FF14',
    fontSize: 14,
    marginTop: 4,
    fontWeight: '600',
  },
});
