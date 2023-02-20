# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 22:44:55 2021

@author: user
"""

# short tutorial about spectral centroid

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

file_path = 'test_files/testBell.wav'
# file_path = 'test_files/testKick.wav'

sr = 44100
n_fft = 2048
hop_length = 1024
range_low = 20
range_high = 10000

# load audio
s, _ = librosa.load( file_path , sr=sr)

# extract centroid
c = librosa.feature.spectral_centroid( s, sr=sr, n_fft=n_fft, hop_length=hop_length)
c = c[0]

# extract bandwidth
b = librosa.feature.spectral_bandwidth( s, sr=sr, n_fft=n_fft, hop_length=hop_length, p=0.1)
b = b[0]

# compute spectrum
p = librosa.stft(s, n_fft=n_fft, hop_length=hop_length)
spectral_magnitude, _ = librosa.magphase(p)
power_spectrum = librosa.amplitude_to_db( np.abs(p), ref=np.max )

# plot
fig , plt_alias =  plt.subplots()
librosa.display.specshow(power_spectrum, hop_length=hop_length, sr=sr, x_axis='time', y_axis='linear', ax=plt_alias)
plt_alias.set_ylim([range_low, range_high])

plt_alias.plot( np.linspace( 0, s.size/sr, c.size ), c , 'g' )
plt_alias.fill_between(np.linspace( 0, s.size/sr, c.size ), c - b, c + b, alpha=0.5)

# %%
# keep only useful part
# plot rms - KEEP ONGOING FIGURE
rms = librosa.feature.rms(S=spectral_magnitude)
rms = rms[0]

plt_alias.plot( np.linspace( 0, s.size/sr, rms.size ), range_high*rms , 'r' )

# %% mask useful part

useful_mask = np.zeros( rms.size )
useful_mask[ rms > 0.005 ] = 1
useful_mask = useful_mask.astype(int)

# keep useful spectrum
useful_spectrum = power_spectrum[:,useful_mask == 1]

fig , plt_alias =  plt.subplots()
librosa.display.specshow(useful_spectrum, hop_length=hop_length, sr=sr, x_axis='time', y_axis='linear', ax=plt_alias)
plt_alias.set_ylim([range_low, range_high])

# keep useful centroid
useful_centroid = c[ useful_mask == 1 ]