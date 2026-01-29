# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 09:27:06 2026

@author: msedo
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

file_path = r"C:\Users\msedo\Documents\SEEG CCEPs\1466569\Gara Castro Mendez-20170424_145332-20250702_103658.edf"

raw = mne.io.read_raw_edf(file_path, preload=True, verbose='info')

meas_date = raw.info['meas_date']
print(f"Measurement Date and Time: {meas_date}")

sfreq = raw.info['sfreq']
print(f"Sampling Frequency: {sfreq} Hz")

channel_names = raw.info['ch_names']
print("Channel Names:", channel_names)

num_channels = len(channel_names)
print(f"Number of Channels: {num_channels}")
highpass = raw.info['highpass']
lowpass = raw.info['lowpass']
print(f"High-Pass Filter: {highpass} Hz")
print(f"Low-Pass Filter: {lowpass} Hz")


# -------------------------- Annotations excel --------------------------------

df = pd.read_excel(r"C:\Users\msedo\Documents\SEEG CCEPs\1466569\Gara Castro Mendez-20170424_145332-20250702_103729.xlsx")


df["Begin"] = pd.to_datetime(df["Begin"], dayfirst=True).dt.tz_localize("UTC")
df["End"]   = pd.to_datetime(df["End"], dayfirst=True).dt.tz_localize("UTC")

meas_date = pd.to_datetime(raw.info["meas_date"], utc=True)

# Keep only annotations that start after the EEG measurement date

df_valid = df[df["Begin"] >= meas_date].copy()

print(f"Kept {len(df_valid)} annotations after meas_date")

# Compute onset (in seconds from meas_date)
df_valid["onset"] = (df_valid["Begin"] - meas_date).dt.total_seconds()

# Compute duration (in seconds)
df_valid["duration"] = (df_valid["End"] - df_valid["Begin"]).dt.total_seconds()


descriptions = (
    df_valid["Text"].astype(str).values
    if "Text" in df_valid.columns
    else ["ANNOT"] * len(df_valid)
)


# Ensure Text is string (important!)
df_valid["Text"] = df_valid["Text"].astype(str)

# Replace '-1' with NaN, then forward-fill
df_valid["Text"] = df_valid["Text"].replace("-1", np.nan).ffill()


if pd.isna(df_valid["Text"].iloc[0]):
    raise ValueError("First annotation Text is '-1' — no previous annotation to inherit from.")

descriptions = df_valid["Text"].values


# Sort annotations by onset just in case
df_valid = df_valid.sort_values("onset").reset_index(drop=True)

# Keep only the first occurrence of each onset
df_valid = df_valid.drop_duplicates(subset="onset", keep="first").reset_index(drop=True)

# Now create MNE Annotations
annotations = mne.Annotations(
    onset=df_valid["onset"].values,
    duration=df_valid["duration"].values,
    description=df_valid["Text"].values,
    orig_time=meas_date,
)

raw.set_annotations(annotations)

raw.plot(duration=5.0, n_channels=20, scalings='auto', show=True)


raw.notch_filter(freqs=[50.0], method='fir', fir_design='firwin')

raw.plot(duration=5.0, n_channels=20, scalings='auto', show=True)

raw_filtered = raw.copy().filter(1., 30., fir_design='firwin')

raw_filtered.plot(duration=5.0, n_channels=20, scalings='auto', show=True)


# Definition of potential Broca (B) and Wernicke (W) channels

pot_B_chs = ["BIA"]
pot_W_chs = ["WCP","TBP","TBA","H","TIA","TP","TOB","HP","TIP","WC","PT","A","AL","TOI","TOS","TIAA","HA","TOM","TIM","TBO","TBM","T2O"]






