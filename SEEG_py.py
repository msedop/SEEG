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

raw = mne.io.read_raw_edf(file_path, preload=False, verbose='info')

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


# Annotations excel

df = pd.read_excel(
    r"C:\Users\msedo\Documents\SEEG CCEPs\1466569\Gara Castro Mendez-20170424_145332-20250702_103729.xlsx"
)


df["Begin"] = pd.to_datetime(df["Begin"], dayfirst=True).dt.tz_localize("UTC")
df["End"]   = pd.to_datetime(df["End"], dayfirst=True).dt.tz_localize("UTC")

meas_date = pd.to_datetime(raw.info["meas_date"], utc=True)





