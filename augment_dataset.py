"""
GymForm AI — Data Augmentation Script
=====================================
Takes the original dataset_fullbody.csv and creates an augmented version
with 3 techniques:
  1. Mirror L/R angles (swap left/right) → doubles the data
  2. Gaussian noise (±3°) → creates realistic variations
  3. Class balancing via oversampling minority classes

Run: python augment_dataset.py
Output: dataset_augmented.csv (original file is NOT modified)
"""

import pandas as pd
import numpy as np
from collections import Counter

# ── Config ──────────────────────────────────────────────────────────
INPUT_CSV = 'dataset_fullbody.csv'
OUTPUT_CSV = 'dataset_augmented.csv'
NOISE_STD = 3.0          # degrees of Gaussian noise
NOISE_COPIES = 3          # how many noisy copies per original row
RANDOM_SEED = 42
# ────────────────────────────────────────────────────────────────────

np.random.seed(RANDOM_SEED)


def mirror_lr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Swap left/right angle columns to simulate a mirrored view.
    This doubles the dataset and helps the model generalize
    to people facing either direction.
    """
    mirrored = df.copy()
    # Swap left ↔ right for each joint pair
    mirrored['l_elbow'], mirrored['r_elbow'] = df['r_elbow'].values, df['l_elbow'].values
    mirrored['l_shoulder'], mirrored['r_shoulder'] = df['r_shoulder'].values, df['l_shoulder'].values
    mirrored['l_hip'], mirrored['r_hip'] = df['r_hip'].values, df['l_hip'].values
    mirrored['l_knee'], mirrored['r_knee'] = df['r_knee'].values, df['l_knee'].values
    return mirrored


def add_gaussian_noise(df: pd.DataFrame, std: float = 3.0, n_copies: int = 3) -> pd.DataFrame:
    """
    Add Gaussian noise (±std degrees) to angle features.
    Simulates natural body variation and measurement jitter.
    Angles are clipped to [0, 360] to stay physically valid.
    """
    angle_cols = ['l_elbow', 'r_elbow', 'l_shoulder', 'r_shoulder',
                  'l_hip', 'r_hip', 'l_knee', 'r_knee']
    
    noisy_copies = []
    for _ in range(n_copies):
        noisy = df.copy()
        noise = np.random.normal(0, std, size=(len(noisy), len(angle_cols)))
        noisy[angle_cols] = noisy[angle_cols].values + noise
        # Clip to valid angle range
        noisy[angle_cols] = noisy[angle_cols].clip(0.0, 360.0)
        noisy_copies.append(noisy)
    
    return pd.concat(noisy_copies, ignore_index=True)


def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oversample minority classes to match the majority class count.
    Uses random duplication with replacement.
    """
    class_counts = df['label'].value_counts()
    max_count = class_counts.max()
    
    balanced_parts = []
    for label, count in class_counts.items():
        class_df = df[df['label'] == label]
        if count < max_count:
            # Oversample: randomly duplicate rows to reach max_count
            extra = class_df.sample(n=max_count - count, replace=True, random_state=RANDOM_SEED)
            class_df = pd.concat([class_df, extra], ignore_index=True)
        balanced_parts.append(class_df)
    
    return pd.concat(balanced_parts, ignore_index=True)


def main():
    # Load original dataset
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Original: {len(df)} rows")
    print(f"  Class distribution:")
    for label, count in sorted(df['label'].value_counts().items()):
        print(f"    {label}: {count}")
    
    # Step 1: Balance classes first (before augmentation)
    print(f"\n[1/3] Balancing classes...")
    df_balanced = balance_classes(df)
    print(f"  After balancing: {len(df_balanced)} rows")
    
    # Step 2: Mirror L/R
    print(f"\n[2/3] Mirroring L/R angles...")
    df_mirrored = mirror_lr(df_balanced)
    df_with_mirrors = pd.concat([df_balanced, df_mirrored], ignore_index=True)
    print(f"  After mirroring: {len(df_with_mirrors)} rows")
    
    # Step 3: Add Gaussian noise
    print(f"\n[3/3] Adding Gaussian noise (std={NOISE_STD}°, {NOISE_COPIES} copies)...")
    df_noisy = add_gaussian_noise(df_with_mirrors, std=NOISE_STD, n_copies=NOISE_COPIES)
    df_final = pd.concat([df_with_mirrors, df_noisy], ignore_index=True)
    print(f"  After noise: {len(df_final)} rows")
    
    # Shuffle
    df_final = df_final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Save
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved augmented dataset to {OUTPUT_CSV}")
    print(f"   Total rows: {len(df_final)} (was {len(df)})")
    print(f"   Expansion: {len(df_final)/len(df):.1f}x")
    print(f"\n  Final class distribution:")
    for label, count in sorted(df_final['label'].value_counts().items()):
        print(f"    {label}: {count}")


if __name__ == '__main__':
    main()
