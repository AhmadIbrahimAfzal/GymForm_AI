import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

const toneColor = {
  good: '#39FF14',
  bad: '#FF4444',
  neutral: '#FFD700',
  default: '#00D2FF',
} as const;

interface Props {
  label: string;
  value: string;
  tone?: keyof typeof toneColor;
}

export function StatCard({ label, value, tone = 'default' }: Props) {
  return (
    <View style={styles.card}>
      <Text style={styles.label}>{label.toUpperCase()}</Text>
      <Text style={[styles.value, { color: toneColor[tone] }]} numberOfLines={1}>
        {value}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    flex: 1,
    margin: 4,
    paddingVertical: 10,
    paddingHorizontal: 8,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(20, 20, 25, 0.85)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.12)',
  },
  label: {
    color: 'rgba(255, 255, 255, 0.5)',
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 1.2,
  },
  value: {
    fontSize: 18,
    fontWeight: '800',
    marginTop: 3,
  },
});
