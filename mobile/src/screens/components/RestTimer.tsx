import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';

interface Props {
  restSeconds: number;
  onRestFinished: () => void;
}

export function RestTimer({ restSeconds, onRestFinished }: Props) {
  const [secondsLeft, setSecondsLeft] = useState(restSeconds);

  useEffect(() => {
    if (secondsLeft <= 0) {
      onRestFinished();
      return;
    }
    const timer = setInterval(() => {
      setSecondsLeft((s) => s - 1);
    }, 1000);
    return () => clearInterval(timer);
  }, [secondsLeft, onRestFinished]);

  return (
    <View style={styles.overlay}>
      <View style={styles.card}>
        <Text style={styles.label}>REST TIME</Text>
        <Text style={styles.time}>{secondsLeft}s</Text>

        <TouchableOpacity
          style={styles.btn}
          activeOpacity={0.8}
          onPress={onRestFinished}
        >
          <Text style={styles.btnText}>SKIP REST</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFill,
    backgroundColor: 'rgba(10, 10, 10, 0.85)',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  card: {
    backgroundColor: 'rgba(20, 20, 25, 0.95)',
    paddingVertical: 32,
    paddingHorizontal: 40,
    borderRadius: 24,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#39FF14',
  },
  label: {
    color: 'rgba(255, 255, 255, 0.6)',
    fontSize: 14,
    fontWeight: '700',
    letterSpacing: 2,
  },
  time: {
    color: '#39FF14',
    fontSize: 54,
    fontWeight: '900',
    marginVertical: 12,
  },
  btn: {
    backgroundColor: '#39FF14',
    paddingVertical: 12,
    paddingHorizontal: 28,
    borderRadius: 12,
    marginTop: 12,
  },
  btnText: {
    color: '#000000',
    fontSize: 14,
    fontWeight: '800',
  },
});
