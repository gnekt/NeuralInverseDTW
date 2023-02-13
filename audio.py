import librosa
import librosa.filters
import math
import numpy as np
import scipy

ref_level_db = 20
preemphasis_ref = 0.97
sample_rate = 22050
num_mels=80
n_fft=1024
hop_length = 256
win_length = 1024
min_level_db = -150

def preemphasis(x):
  return scipy.signal.lfilter([1, -preemphasis_ref], [1], x)


def inv_preemphasis(x):
  return scipy.signal.lfilter([1], [1, -preemphasis_ref], x)

def spectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(np.abs(D)) - ref_level_db
  return _normalize(S)

def melspectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(_linear_to_mel(np.abs(D))) - ref_level_db
  rev = from_taco1_to_taco2_melspectrogram(_normalize(S))
  return _normalize(S)

def from_taco1_to_taco2_melspectrogram(y):
  S = _denormalize(y)
  mel = _db_to_amp(S) 
  return mel

def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(sample_rate * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)



def _stft(y):
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

# Conversions:

_mel_basis = None

def _linear_to_mel(spectrogram):
  global _mel_basis
  if _mel_basis is None:
    _mel_basis = _build_mel_basis()
  return np.dot(_mel_basis, spectrogram)

def _build_mel_basis():
  return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels)

def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
  return np.power(10.0, x * 0.05)

def _normalize(S):
  return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def _denormalize(S):
  return (np.clip(S, 0, 1) * -min_level_db) + min_level_db