# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 09:27:06 2026

@author: msedo
"""

# Libraries import

import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import itertools
import re

# Function definitions


def create_bipolar_by_prefix_all(raw, prefixes, sep='-'):
    """
    Create bipolar channels for ALL possible channel pairs
    sharing the same prefix (e.g. TIP1-TIP4).

    Parameters
    ----------
    raw : mne.io.Raw
        Original raw object (monopolar SEEG).
    prefixes : list of str
        Channel name prefixes (e.g. ['TIP', 'TIA']).
    sep : str
        Separator for bipolar channel names.

    Returns
    -------
    raw_bipolar : mne.io.Raw
        Raw object containing bipolar channels only.
    """

    sfreq = raw.info['sfreq']
    meas_date = raw.info['meas_date']

    bipolar_data = []
    bipolar_names = []
    used_pairs = set()

    for prefix in prefixes:
        # Match prefix + digits ONLY (e.g. TIP12)
        pattern = re.compile(rf"^{prefix}\d+$")

        chs = [ch for ch in raw.ch_names if pattern.match(ch)]

        if len(chs) < 2:
            continue

        for ch1, ch2 in itertools.combinations(chs, 2):
            pair_id = tuple(sorted((ch1, ch2)))
            if pair_id in used_pairs:
                continue

            used_pairs.add(pair_id)

            data = raw.get_data(picks=[ch1]) - raw.get_data(picks=[ch2])
            bipolar_data.append(data[0])
            bipolar_names.append(f"{ch1}{sep}{ch2}")

    if not bipolar_data:
        raise RuntimeError("No bipolar channels created.")

    bipolar_data = np.vstack(bipolar_data)

    info = mne.create_info(
        ch_names=bipolar_names,
        sfreq=sfreq,
        ch_types='seeg'
    )

    raw_bipolar = mne.io.RawArray(bipolar_data, info)
    raw_bipolar.set_meas_date(meas_date)

    if raw.annotations is not None:
        raw_bipolar.set_annotations(raw.annotations)

    return raw_bipolar



# ------------------------------- MAIN CODE -----------------------------------

file_path = r"C:\Users\msedo\Documents\SEEG CCEPs\1466569\Gara Castro Mendez-20170423_125328-20250702_103156.edf"

raw = mne.io.read_raw_edf(file_path, preload=True, verbose='info')

# remove 'EEG' prefix from channels
raw.rename_channels(
    lambda ch: ch.replace("EEG ", "")
)


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

df = pd.read_excel(r"C:\Users\msedo\Documents\SEEG CCEPs\1466569\Gara Castro Mendez-20170423_125328-20250702_103338.xlsx")


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

# Replace '-1' event descriptions with last stimulation description

# Ensure Text is string
df_valid["Text"] = df_valid["Text"].astype(str)

# Replace '-1' with NaN, then forward-fill
df_valid["Text"] = df_valid["Text"].replace("-1", np.nan).ffill()


if pd.isna(df_valid["Text"].iloc[0]):
    raise ValueError("First annotation Text is '-1' — no previous annotation to inherit from.")

descriptions = df_valid["Text"].values

# In case we have duplicate onset times:

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

# ------------------- Data visualisation and filtering -------------------------

raw.plot(duration=5.0, n_channels=20, scalings='100e-6', show=True)

raw_filtered = raw.copy().filter(1., 80., fir_design='firwin')

raw_filtered.plot(duration=5.0, n_channels=20, scalings='100e-6', show=True)


# --------------------- Setting Bipolar Montage -------------------------------

# Definition of potential Broca (B) and Wernicke (W) channels

pot_B_chs = ["BIA"]
pot_W_chs = ["WCP","TBP","TBA","H","TIA","TP","TOB","HP","TIP","WC","PT","A","AL",
             "TOI","TOS","TIAA","HA","TOM","TIM","TBO","TBM","T2O"]



raw_bipolar_W_all = create_bipolar_by_prefix_all(
    raw_filtered,
    prefixes=pot_W_chs
)

raw_bipolar_W_all.plot(
    duration=2.0,
    n_channels=20,
    scalings='auto',
    show=True
)


raw_bipolar_B_all = create_bipolar_by_prefix_all(
    raw_filtered,
    prefixes=pot_B_chs
)

raw_bipolar_B_all.plot(
    duration=2.0,
    n_channels=20,
    scalings='auto',
    show=True
)